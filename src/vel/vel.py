import os
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import itslive
from ticoi.utils import find_granule_by_point
from ticoi.core import ticoi_one_pixel
import pickle


def itslive_collect_info(velocities: list):
    df = pd.DataFrame()
    for key in [key for key in velocities[0].keys() if key != "time_series"]:
        df[key] = [v[key] for v in velocities]

    return df


class Filter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def velocity_nans(self):
        return self.df.loc[self.df[["vx", "vy", "v"]].notna().all(axis=1)]

    def baselines(self, lim: tuple[int, int]):
        return self.df.loc[
            (self.df.date_dt >= pd.to_timedelta(lim[0], "D")) & (self.df.date_dt <= pd.to_timedelta(lim[1], "D"))
            ]

    def satellites(self, sensors: list[str]):
        if "S1" in sensors:
            sensors += ["S1A", "S1B", "S1C", "S1D"]

        if "S2" in sensors:
            sensors += ["S2A", "S2B", "S2C", "S2D"]

        return self.df.loc[~(self.df.mission_img1 + self.df.satellite_img1).isin(sensors)]

    def removeOutliers(self, lim: tuple[int, int] = None, sensors: list[str] = None):
        fdf = self.df.copy()

        if lim:
            fdf = Filter(df=fdf).baselines(lim=lim)

        if sensors:
            fdf = Filter(df=fdf).satellites(sensors=sensors)

        vmin = fdf.v.describe()["25%"] - 1.5 * (fdf.v.describe()["75%"] - fdf.v.describe()["25%"])
        vmax = fdf.v.describe()["75%"] + 1.5 * (fdf.v.describe()["75%"] - fdf.v.describe()["25%"])

        return self.df.loc[(self.df.v >= vmin) & (self.df.v <= vmax)]


class Select:
    def __init__(self, df: pd.DataFrame, tlim: tuple[str, str]):
        self.df = df
        self.tlim = tlim

    def min_baseline(self):
        dft = pd.DataFrame(
            index=pd.Index(pd.date_range(self.tlim[0], self.tlim[1]), name="date"),
            data={"date_dt": pd.Timedelta(9999, "D"), "vx": np.nan, "vy": np.nan, "v": np.nan}
        )

        for ind in dft.index:
            df_temp = self.df.loc[
                (self.df.acquisition_date_img1 <= ind) & (self.df.acquisition_date_img2 >= ind)
                ]
            if not df_temp.empty:
                dft.loc[ind] = df_temp.loc[
                    df_temp.date_dt == df_temp.date_dt.min(), ["date_dt", "vx", "vy", "v"]
                ].mean()

        return dft


def itslive_filter_point_timeseries(
        DF: pd.DataFrame,
        filepaths: list[os.PathLike],
        tlim: tuple[str, str],
        **kwargs
):
    max_baseline = kwargs.get("max_baseline", 120)

    for filepath in tqdm(filepaths, total=len(filepaths), desc="Reading and filtering ITS_LIVE time series"):
        ds = xr.open_dataset(filepath, engine="netcdf4")

        df = ds.sortby("mid_date").sel(
            mid_date=slice(
                pd.to_datetime(tlim[0]) - pd.Timedelta(max_baseline, "D"),
                pd.to_datetime(tlim[1]) + pd.Timedelta(max_baseline, "D")
            )
        ).to_dataframe().drop(columns=["x", "y"])

        # Default filter out nan values and the max baselines.
        df = Filter(df=df).velocity_nans()
        df = Filter(df=df).baselines(lim=(0, max_baseline))

        # Optional filters
        if kwargs.get("filter_sensors", False):
            df = Filter(df=df).satellites(sensors=kwargs.get("filter_sensors"))

        if kwargs.get("filter_outliers", False):
            df = Filter(df=df).removeOutliers(
                lim=kwargs.get("outlier_baseline_lim", (0, max_baseline)),
                sensors=kwargs.get("outlier_sensors", None)
            )

        df = Select(df=df, tlim=tlim).min_baseline()

        df = df.set_index(pd.MultiIndex.from_product([[int(filepath[-7:-3])], df.index], names=["distance", "date"]))

        DF.loc[df.index] = df

    return DF


def itslive_load_points(
        points: list[tuple[float, float]],
        indices: list,
        dirpath: str | os.PathLike,
        download=False
):
    """
    A function that finds/downloads ITS_LIVE velocities of point coordinates.
    :param points: Coordinate pairs as [(lon, lat), ...]
    :param indices: Indices of points along profile.
    :param dirpath: Directory path where ITS_LIVE velocities are saved.
    :param download: Whether to read velocities from the directory path or download from ITS_LIVE.
    :return : Dataframe with ITS_LIVE point info and a list of ITS_LIVE velocity files.
    """
    itslive_files = [os.path.join(dirpath, f"{os.path.basename(dirpath)}_{i:04}.nc") for i in indices]
    if not all([os.path.exists(path) for path in itslive_files]):
        download = True

    if not download:
        def read_pickle(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            return data

        velocities_itslive = [
            read_pickle(os.path.join(dirpath, f"{os.path.basename(dirpath)}_{i:04}")) for i in indices
        ]
    else:
        if len(indices) != len(points):
            raise ValueError("Indices must be of equal length to points")

        velocities_itslive = itslive.velocity_cubes.get_time_series(
            points=points,
            variables=["acquisition_date_img1", "acquisition_date_img2"]
        )
        for i, v in tqdm(zip(indices, velocities_itslive), total=len(indices), desc="Downloading ITS_LIVE data"):
            with open(os.path.join(dirpath, f"{os.path.basename(dirpath)}_{i:04}"), 'wb') as file:
                pickle.dump({key: v[key] for key in v.keys() if key != "time_series"}, file)

            v["time_series"].to_netcdf(os.path.join(dirpath, f"{os.path.basename(dirpath)}_{i:04}.nc"))

    return itslive_collect_info(velocities=velocities_itslive), itslive_files


def itslive_decompose_seasonal(ds, var="v", tlim: tuple[str, str] = None, plot=False):
    dvar = f"d{var}_seasonal"
    ds_new = ds.assign({dvar: xr.full_like(ds[var], fill_value=np.nan)})

    for distance in ds.distance.values:
        ts = ds.sel(distance=distance).to_pandas()[[var]]  # ds.distance[int(len(ds.distance)/2)].item()
        ts_new = pd.DataFrame(index=ts.index, columns=[dvar], data=ts[var].values).fillna(np.nan)

        if tlim:
            ts = ts.loc[(ts.index >= tlim[0]) & (ts.index <= tlim[1])]
        ts = ts.asfreq("D")
        ts = ts.interpolate(method="time")
        ts = ts.loc[~ts[var].isna()]

        try:
            res = STL(ts, period=365, seasonal=36501, seasonal_deg=0, robust=True).fit()
        except IndexError:
            try:
                res = seasonal_decompose(ts, model="additive", period=365)
            except ValueError:
                res = None

        if plot:  # Compare STL and seasonal_decompose - and switch if necessary.
            res.plot()
            seasonal_decompose(ts, model="additive", period=365).plot()

        for date in pd.DatetimeIndex(pd.date_range("2024-01-01", "2024-12-31", freq="1D")).date:
            if res:
                md = res.seasonal.loc[
                    (res.seasonal.index.month == date.month) & (res.seasonal.index.day == date.day)
                ].mean()  # Compute mean seasonal value for month/day
                ts_new.loc[(ts_new.index.month == date.month) & (ts_new.index.day == date.day), dvar] = ts_new.loc[
                    (ts_new.index.month == date.month) & (ts_new.index.day == date.day), dvar
                ] - md

        ds_new[dvar].loc[ds_new.distance == distance] = ts_new[dvar].to_xarray()

    return ds_new


def ticoi_point_timeseries(
        point: tuple[float, float],
        **kwargs
):
    df = pd.DataFrame()
    n = 0
    while len(df) == 0 and n < 5:
        n += 1
        try:
            _, _, df = ticoi_one_pixel(
                cube_name=find_granule_by_point([point[0], point[1]]),
                i=point[0],
                j=point[1],
                save=kwargs.get("save", False),
                path_save=kwargs.get("path_save", "To_fill_if_save_True"),
                show=kwargs.get("show", False),
                option_visual=kwargs.get("option_visual", ["obs_magnitude", "invertvv", "inverpvv", "quality_metrics"]),
                load_kwargs={
                    "pick_date": ("1984-01-01", "2026-01-01"),
                    "buffer": [point[0], point[1], 0.01],
                    "chunks": "auto"
                } | kwargs.get("load_kwargs", {}),
                load_pixel_kwargs=kwargs.get("load_pixel_kwargs", {"visual": True}),
                preData_kwargs=kwargs.get("preData_kwargs", {"delete_outliers": {"median_angle": 45}}),
                inversion_kwargs=kwargs.get("inversion_kwargs", {
                    "coef": kwargs.get("coef", 100),
                    "result_quality": kwargs.get("result_quality", ["X_contribution"]),
                    "visual": True
                }),
                interpolation_kwargs={"result_quality": kwargs.get("result_quality", ["X_contribution"])}
            )
        except (KeyError, zarr.errors.GroupNotFoundError) as e:
            df = pd.DataFrame()

    return df


def ticoi_load_points(
        points: list[tuple[float, float]],
        indices: list,
        dirpath: str | os.PathLike,
        rgi_l_id: str,
        download=False,
        **kwargs
):
    ticoi_files = [os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv") for i in indices]
    # if not all([os.path.exists(path) for path in ticoi_files]):
    #     download = True

    if download:
        for i, point in tqdm(
                zip(indices, points), total=len(indices),
                desc="Downloading TICOI for centreline {} on volcano {}".format(
                    dirpath.split('\\')[-3], dirpath.split('\\')[-4]
                )
        ):
            df = ticoi_point_timeseries(point=point, **kwargs)
            if len(df) == 0:
                ticoi_files.remove(os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv"))
                continue

            df.to_csv(os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv"), index=False)
    else:
        for i, point in tqdm(
                zip(indices, points), total=len(indices),
                desc="Downloading TICOI for centreline {} on volcano {}".format(
                    dirpath.split("\\")[-3], dirpath.split("\\")[-4]
                )
        ):
            if not os.path.exists(os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv")):
                df = ticoi_point_timeseries(point=point, **kwargs)
                if len(df) == 0:
                    ticoi_files.remove(os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv"))
                    continue
                df.to_csv(os.path.join(dirpath, f"{rgi_l_id}_{i:04}_ticoi.csv"), index=False)

    return ticoi_files


def ticoi_combine_timeseries(filepaths: list[str | os.PathLike], distance: list[float], tlim: tuple[str, str]):
    tlim = (pd.to_datetime(tlim[0]) - pd.Timedelta(100, "D"),
            pd.to_datetime(tlim[1]) + pd.Timedelta(100, "D"))
    dfv = pd.DataFrame()

    for filepath, d in tqdm(zip(filepaths, distance), total=len(filepaths), desc="Assembling TICOI timeseries"):
        df = pd.read_csv(filepath)
        df[["date1", "date2"]] = df[["date1", "date2"]].apply(pd.to_datetime)
        df["date"] = df.date1 + (df.date2 - df.date1)/2

        df = df.loc[(df["date"] >= tlim[0]) & (df["date"] <= tlim[1])]

        df["v"] = np.sqrt(df.vx**2 + df.vy**2)
        df["distance"] = d

        if len(dfv) == 0:
            dfv = df.copy()
        else:
            dfv = pd.concat([dfv, df])

    return dfv.set_index(["distance", "date"])
