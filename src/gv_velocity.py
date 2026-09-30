import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import multiprocessing as mp

import tools.tools as tools
import geo.geo as geo
import gvf.gvf as gvf
import vel.vel as vel
import era.era as era
import tkn.tkn as tkn

# Set the root directory as the project directory.
dir_root = tools.set_root(path=os.getcwd(), name="gv_velocity")
# Data directories.
dir_data = os.path.join(dir_root, "data")
dir_data_proc = os.path.join(dir_root, "data_processed")
dir_figs = os.path.join(dir_root, "figs")

RGI_v = "RGI2000-v7.0"

# Parallelise the code so many glacier centrelines can be run simultaneously
n_parallel = 12
if n_parallel >= mp.cpu_count():
    n_parallel = mp.cpu_count() - 2

ids = {
    300260: {"10-06188": ["10-06926"]},
    300261: {"10-06190": ["10-06931"]},
    312070: {"01-06836": ["01-09985"]},
    313020: {
        "01-03504": ["01-04946"],
        "01-03581": ["01-05053", "01-05054"],
        "01-03582": ["01-05055", "01-05057"],  # fix in manuscript
        "01-03587": ["01-05064"],
        "01-03588": ["01-05066", "01-05067"]  # fix in manuscript
    },
    313040: {
        "01-02593": ["01-03640"],
        "01-02602": ["01-03660"],
        # "01-02603": ["01-03664"],
        "01-02605": ["01-03668"],
        "01-02607": ["01-03671"],
        "01-02613": ["01-03681"],
    },
    315020: {
        "01-05906": ["01-08691"],
        "01-06010": ["01-08823"],
        "01-06340": ["01-09325"]
    },
    315030: {
        "01-16317": ["01-24546"],
        "01-16437": ["01-24732"]
    },
    320160: {"02-08538": ["02-12164"]},
    # 321030: {},
    # 357010: {
    #     "17-25773": ["17-30360", "17-30361"],
    #     "17-25826": ["17-30418", "17-30423", "17-30424"],
    # },
    357020: {"17-25664": ["17-30220"]},
    357040: {},
    # 358057: {"17-13041": ["17-15696"]},
    # 358060: {
    #     "17-12034": ["17-14340"],
    #     "17-12163": ["17-14504"],
    #     "17-12185": ["17-14535"],
    #     "17-12247": ["17-14656"],
    #     "17-12319": ["17-14755"]
    # },
    358062: {"17-14377": ["17-17226"]},  # Fix in manuscript
    358063: {"17-01845": ["17-02155"]},
    371090: {},
    372020: {"06-00421": ["06-00691"]},
    # 372030: {
    #     "06-00406": ["06-00660"],
    #     "06-00407": ["06-00666"],
    #     "06-00436": ["06-00716"],
    #     "06-00437": ["06-00720", "06-00721"]
    # },
    373010: {
        #"06-00224": ["06-00315"],  # Remove from manuscript
        "06-00225": ["06-00315"],
        "06-00227": ["06-00331"] #"06-00330"]  #Add to manuscript?
    },
    373030: {
        "06-00221": ["06-00303"],
        "06-00375": ["06-00585", "06-00584"]  # Add to manuscript?
    },
    # 373050: {"06-00373": ["06-00575"]},
    374010: {
        "06-00250": ["06-00370"],
        "06-00376": ["06-00601"]
    },
}

plot_kwargs = {
    "10-06926": {"ylim": ("2017-01-01", "2025-01-01")},
    "10-06931": {"ylim": ("2017-01-01", "2025-01-01")},
    "01-03664": {"xlim": (0, 13)},
    "06-00331": {"xlim": (0, 35)}
}

# The time period of interest - specify the time period for the centreline ID.
tlims = {
    "01-09985": ("2017-01-01", "2025-01-01"),
    "01-05053": ("2000-01-01", "2025-01-01"),
    "01-05054": ("2000-01-01", "2025-01-01"),
    "01-05067": ("2000-01-01", "2025-01-01"),
    "01-03640": ("1995-01-01", "2025-01-01"),
    "01-03664": ("1995-01-01", "2025-01-01"),
    "01-03681": ("1995-01-01", "2025-01-01"),
    "01-08691": ("1995-01-01", "2025-01-01"),
    "01-08823": ("1995-01-01", "2025-01-01"),
    "01-09325": ("1995-01-01", "2025-01-01"),
    "01-24546": ("1985-01-01", "2025-01-01"),
    "01-24732": ("1995-01-01", "2025-01-01"),
    "17-14504": ("2000-01-01", "2025-01-01"),
    "17-30423": ("1985-01-01", "2025-01-01"),
    "17-30424": ("1985-01-01", "2025-01-01"),
    "17-30220": ("1985-01-01", "2025-01-01"),
    "17-29154": ("1985-01-01", "2025-01-01"),
    "17-29156": ("1985-01-01", "2025-01-01"),
    "17-29164": ("1985-01-01", "2025-01-01"),
    "17-29166": ("1985-01-01", "2025-01-01"),
    "17-29172": ("1985-01-01", "2025-01-01"),
    "17-29176": ("1985-01-01", "2025-01-01"),
}
tlim_default = ("2015-01-01", "2025-01-01")

# Default parameters
max_segment_length = 200
stats_keys = ["dv_mean", "dv_mean_vec", "dv_median", "dv_seasonal", "dv_seasonal_vec"]

itslive_kwargs = {"itslive_name": "default"}
# itslive_kwargs = {
#     "itslive_name": "filtered",
#     "max_baseline": 120,
#     "filter_sensors": None,  # e.g. ["S1", "S1A, "L8", "S2B", ...]
#     "filter_outliers": True,
#     "outlier_baseline_lim": (0, 45),
#     "outlier_exclude_sensors": ["S1"],
# }

ticoi_kwargs = {"ticoi_name": "default"}
# ticoi_kwargs = {
#     "ticoi_name": "test_01",
#     "option_visual": ["obs_magnitude", "invertvv", "inverpvv", "quality_metrics"],
#     "load_kwargs": {
#         "pick_temp_bas": [1, 32],
#         "pick_sensor": ["Landsat-5", "Landsat-7", "Landsat-8", "Landsat-9", "Sentinel-2"]
#     },
#     "inversion_kwargs": {
#         "coef": 100,
#         "result_quality": ["X_contribution"],
#         "visual": True,
#         "detect_temporal_decorrelation": True,
#         "iteration": True,  # Allow the inversion process to make several iterations
#         "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
#         "threshold_it": 0.1,  # Threshold to test stability of results between each iteration, used to stop the process
#     },
#     "interpolation_kwargs": {
#         "result_quality": ["X_contribution"],
#         "interval_output": 15,
#     }
# }
# ticoi_kwargs = {
#     "ticoi_name": "test_02",
#     "option_visual": ["obs_magnitude", "invertvv", "inverpvv", "quality_metrics"],
#     "load_kwargs": {
#         "pick_temp_bas": [1, 45],
#         "pick_sensor": ["Landsat-5", "Landsat-7", "Landsat-8", "Landsat-9", "Sentinel-2", "Sentinel-1"]
#     },
#     "preData_kwargs": {
#         "delete_outliers": {"median_angle": 45, "iqr": 1.5}
#     },
#     "inversion_kwargs": {
#         "coef": 100,
#         "result_quality": ["X_contribution"],
#         "visual": True,
#         "detect_temporal_decorrelation": True,
#         "iteration": True,  # Allow the inversion process to make several iterations
#         "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
#         "threshold_it": 0.1,  # Threshold to test stability of results between each iteration, used to stop the process
#     },
#     "interpolation_kwargs": {
#         "result_quality": ["X_contribution"],
#         "interval_output": 15,
#     }
# }


class Volcano:
    def __init__(self, gvp_id, radius=10, max_segment_length_m=200, rgi_v="RGI2000-v7.0"):
        self.id = gvp_id
        df = gpd.GeoDataFrame(
            gvf.read_gvp(
                os.path.join(dir_data, "GVP", "GVP_Volcano_List_Holocene.csv")
            )
        )
        self.gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.GeoSeries.from_xy(df["Longitude"], df["Latitude"]),
            crs="EPSG:4326"
        ).loc[df["Volcano Number"] == self.id]
        self.radius = radius
        self.max_segment_length_m = max_segment_length_m
        self.dir_gvp = tools.find_subdirs_within_path(
            path=os.path.join(dir_data_proc, "regional_files", f"{rgi_v}-GV"),
            dirname=f"{self.id}"
        )[0]
        self.rgi_region = self.dir_gvp.split(os.sep)[-3].split("-")[-1]
        self.glaciers = gpd.read_file(os.path.join(self.dir_gvp, f"{self.id}_{float(radius)}km-glaciers.shp"))
        self.centrelines = gpd.read_file(os.path.join(self.dir_gvp, f"{self.id}_{float(radius)}km-centrelines.shp"))
        cls_file = os.path.join(
            self.dir_gvp,
            f"{self.id}_{float(self.radius)}km-centrelines_s{self.max_segment_length_m}.shp"
        )
        if os.path.exists(cls_file):
            self.centrelines_s = gpd.read_file(cls_file)
        else:
            self.centrelines_s = geo.simplify_polylines(gdf=self.centrelines)
            self.centrelines_s.to_file(cls_file)

        dir_csv = os.path.join(dir_data_proc, "regional_files", f"{rgi_v}-V", f"{rgi_v}-V-{self.rgi_region}")
        self.unrest, self.eruptions = gvf.gvp_unrest_eruptions(dir_csv=dir_csv, GVP_id=self.id)


class Glacier:
    def __init__(self, rgi_id, volcano: Volcano):
        self.id = rgi_id
        self.glacier = volcano.glaciers.loc[volcano.glaciers.rgi_id == self.id]
        self.dir_gvp = volcano.dir_gvp
        self.centrelines = volcano.centrelines.loc[volcano.centrelines.rgi_g_id == self.id]
        self.centrelines_s = volcano.centrelines_s.loc[volcano.centrelines_s.rgi_g_id == self.id]

    def centreline(self, rgi_l_id):
        return self.centrelines.loc[self.centrelines.rgi_id == rgi_l_id]

    def centreline_s(self, rgi_l_id):
        return self.centrelines_s.loc[self.centrelines_s.rgi_id == rgi_l_id]

    def dir_centreline(self, rgi_l_id):
        return tools.mkdir_ifnot_exist(
            path_proj=dir_root,
            path=os.path.join(self.dir_gvp, rgi_l_id)
        )


def centreline_velocities_itslive(glacier: Glacier, rgi_l_id: str, tlim=tlim_default, indices=None):
    itslive_path = os.path.join(
        glacier.dir_gvp, f"{rgi_l_id}_itslive_{itslive_kwargs['itslive_name']}_{tlim[0]}_{tlim[1]}.nc"
    )
    itslive_path_new = os.path.join(
        glacier.dir_gvp, f"{rgi_l_id}_itslive_{itslive_kwargs['itslive_name']}_{tlim[0]}_{tlim[1]}_new.nc"
    )
    if os.path.exists(itslive_path):
        with xr.open_dataset(itslive_path, engine="netcdf4") as ds:
            if all(key in ds.keys() for key in stats_keys):
                return ds
            else:
                dsv = ds_stats(ds=ds)
                dsv.to_netcdf(itslive_path_new)

        os.replace(itslive_path_new, itslive_path)

        return dsv
    else:
        centreline = glacier.centreline_s(rgi_l_id=rgi_l_id)

        if centreline.length_m.values[0] < 2.e3:
            return {}

        distance = geo.profile_distance(polyline=centreline) / 1e3
        coordinates = centreline.get_coordinates().values

        if not indices:
            indices = np.arange(len(distance), dtype=int).tolist()
        else:
            distance = distance[indices]
            coordinates = coordinates[indices]

        dfv = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=[indices, pd.date_range(tlim[0], tlim[1])],
                names=["distance", "date"]
            ),
            data={"date_dt": pd.Timedelta(9999, "D"), "vx": np.nan, "vy": np.nan, "v": np.nan}
        )

        df_info, itslive_filelist = vel.itslive_load_points(
            points=coordinates, indices=indices, dirpath=glacier.dir_centreline(rgi_l_id=rgi_l_id)
        )
        dfv = vel.itslive_filter_point_timeseries(
            DF=dfv,
            filepaths=itslive_filelist,
            tlim=tlim,
            **itslive_kwargs
        )
        dfv.index = dfv.index.set_levels(distance, level="distance")
        # dsv = dfv.loc[dfv.date_dt != pd.Timedelta(9999, "D")].to_xarray()

        dsv = dfv.to_xarray()
        dsv.to_netcdf(os.path.join(glacier.dir_gvp, f"{rgi_l_id}_itslive_filtered_{tlim[0]}_{tlim[1]}.nc"))
        dsv = ds_stats(ds=dsv)
        dsv.to_netcdf(os.path.join(glacier.dir_gvp, f"{rgi_l_id}_itslive_filtered_{tlim[0]}_{tlim[1]}.nc"))
        return dsv


def centreline_velocities_ticoi(glacier: Glacier, rgi_l_id: str, tlim=tlim_default, indices=None):
    ticoi_path_old = os.path.join(
        glacier.dir_gvp, f"{rgi_l_id}_ticoi_{ticoi_kwargs['ticoi_name']}.nc"
    )
    ticoi_path = os.path.join(
        glacier.dir_gvp, f"{rgi_l_id}_ticoi_{ticoi_kwargs['ticoi_name']}_{tlim[0]}_{tlim[1]}.nc"
    )
    ticoi_path_new = os.path.join(
        glacier.dir_gvp, f"{rgi_l_id}_ticoi_{ticoi_kwargs['ticoi_name']}_{tlim[0]}_{tlim[1]}_new.nc"
    )
    if os.path.exists(ticoi_path_old):
        with xr.open_dataset(ticoi_path_old, engine="netcdf4") as ds:
            date_min = pd.Timestamp(ds.date.min().item())
            date_max = pd.Timestamp(ds.date.max().item())
            if date_min - pd.Timestamp(date_min.year, 1, 1) > pd.Timedelta(180, "D"):
                date_min = pd.Timestamp(date_min.year + 1, 1, 1)
            else:
                date_min = pd.Timestamp(date_min.year, 1, 1)

            if date_max - pd.Timestamp(date_max.year, 1, 1) > pd.Timedelta(180, "D"):
                date_max = pd.Timestamp(date_max.year + 1, 1, 1)
            else:
                date_max = pd.Timestamp(date_max.year, 1, 1)

        os.replace(
            ticoi_path_old,
            os.path.join(
                glacier.dir_gvp,
                f"{rgi_l_id}_ticoi_{ticoi_kwargs['ticoi_name']}_{date_min.strftime('%Y-%m-%d')}_{date_max.strftime('%Y-%m-%d')}.nc"
            )
        )

    if os.path.exists(ticoi_path):
        with xr.open_dataset(ticoi_path, engine="netcdf4") as ds:
            if all(key in ds.keys() for key in stats_keys):
                return ds
            else:
                dsv = ds_stats(ds=ds)
                dsv.to_netcdf(ticoi_path_new)

        os.replace(ticoi_path_new, ticoi_path)

        return dsv
    else:
        dirpath = tools.mkdir_ifnot_exist(
            path=os.path.join(glacier.dir_centreline(rgi_l_id=rgi_l_id), "ticoi", ticoi_kwargs['ticoi_name']),
            path_proj=glacier.dir_centreline(rgi_l_id=rgi_l_id)
        )

        centreline = glacier.centreline_s(rgi_l_id=rgi_l_id)

        if centreline.length_m.values[0] < 2.e3:
            return {}

        distance = geo.profile_distance(polyline=centreline) / 1e3
        coordinates = centreline.get_coordinates().values

        if not indices:
            indices = np.arange(len(distance), dtype=int).tolist()
        else:
            distance = distance[indices]
            coordinates = coordinates[indices]

        ticoi_filelist = vel.ticoi_load_points(
            points=coordinates, indices=indices, dirpath=dirpath, rgi_l_id=rgi_l_id, **ticoi_kwargs
        )

        dfv = vel.ticoi_combine_timeseries(filepaths=ticoi_filelist, distance=distance, tlim=tlim)

        dsv = dfv.to_xarray()
        dsv = ds_stats(ds=dsv)
        dsv.to_netcdf(ticoi_path)
        return dsv


def ds_stats(ds, tlim_stats: tuple[str, str] = None, override_vars=False):
    if override_vars:
        ds = ds.drop_vars(stats_keys)

    if "dv_mean" not in list(ds.keys()):
        ds = ds.assign(dv_mean=ds.v - ds.v.mean(dim="date"))
        ds["dv_mean"] = ds["dv_mean"] / abs(ds["dv_mean"]).max(dim="date")

    if "dv_mean_vec" not in list(ds.keys()):
        ds = ds.assign(
            dv_mean_vec=((ds["vx"] - ds["vx"].mean(dim="date")) * ds["vx"].mean(dim="date") +
                         (ds["vy"] - ds["vy"].mean(dim="date")) * ds["vy"].mean(dim="date")) / ds["v"].mean(dim="date")
        )
        ds["dv_mean_vec"] = ds["dv_mean_vec"] / abs(ds["dv_mean_vec"]).max(dim="date")

    if "dv_median" not in list(ds.keys()):
        ds = ds.assign(dv_median=ds.v - ds.v.median(dim="date"))
        ds["dv_median"] = ds["dv_median"] / abs(ds["dv_median"]).max(dim="date")

    if "dv_seasonal" not in list(ds.keys()):
        ds = vel.itslive_decompose_seasonal(ds=ds, var="v", tlim=tlim_stats)
        ds["dv_seasonal"] = ds["dv_seasonal"] - ds["dv_seasonal"].mean(dim="date")
        ds["dv_seasonal"] = ds["dv_seasonal"] / abs(ds["dv_seasonal"]).max(dim="date")

    if "dv_seasonal_vec" not in list(ds.keys()):
        ds = vel.itslive_decompose_seasonal(ds=ds, var="vx", tlim=tlim_stats)
        ds = vel.itslive_decompose_seasonal(ds=ds, var="vy", tlim=tlim_stats)
        ds = ds.assign(
            dv_seasonal_vec=(
                ((ds["dvx_seasonal"] - ds["dvx_seasonal"].mean(dim="date")) * ds["dvx_seasonal"].mean(dim="date") +
                 (ds["dvy_seasonal"] - ds["dvy_seasonal"].mean(dim="date")) * ds["dvy_seasonal"].mean(dim="date")) /
                np.sqrt(ds["dvx_seasonal"].mean(dim="date") ** 2 + ds["dvy_seasonal"].mean(dim="date") ** 2)
            )
        )
        ds["dv_seasonal_vec"] = ds["dv_seasonal_vec"] / abs(ds["dv_seasonal_vec"]).max(dim="date")

    return ds


def run_volcano_glacier_centreline_df(df: pd.DataFrame):
    for i in df.index:
        GVP_id, RGI_G_id, RGI_L_id = df.loc[i]
        print(f"Volcano: {GVP_id}, Glacier: {RGI_G_id}, Centreline: {RGI_L_id}")

        volcano = Volcano(gvp_id=GVP_id)
        glacier = Glacier(rgi_id=RGI_G_id, volcano=volcano)

        if RGI_L_id[-8:] in tlims.keys():
            tlim = tlims[RGI_L_id[-8:]]
        else:
            tlim = tlim_default

        d = era.read_and_process_ERA5(
            filepath=os.path.join(dir_data, "ERA5", f"era5_{GVP_id}.nc")
        )

        if RGI_L_id[-8:] in plot_kwargs.keys():
            kwargs = plot_kwargs[RGI_L_id[-8:]]
            if "ylim" in kwargs.keys():
                ylim = pd.to_datetime(kwargs["ylim"])
                kwargs.pop("ylim")
            else:
                ylim = pd.to_datetime(tlim)
        else:
            kwargs = {}
            ylim = pd.to_datetime(tlim)

        ds_i = centreline_velocities_itslive(glacier=glacier, rgi_l_id=RGI_L_id, tlim=tlim)
        tkn.plot_single(
            ds=ds_i, d_era5=d, df_unrest=volcano.unrest, df_eruptions=volcano.eruptions,
            ylim=ylim,
            filepath=os.path.join(
                dir_figs,
                f"{GVP_id}_{RGI_G_id}_{RGI_L_id[-10:]}_{tlim[0]}-{tlim[1]}_itslive_{itslive_kwargs['itslive_name']}.png"
            ),
            **kwargs
        )
        ds_t = centreline_velocities_ticoi(glacier=glacier, rgi_l_id=RGI_L_id, tlim=tlim)
        tkn.plot_single(
            ds=ds_t, d_era5=d, df_unrest=volcano.unrest, df_eruptions=volcano.eruptions,
            ylim=ylim,
            filepath=os.path.join(
                dir_figs,
                f"{GVP_id}_{RGI_G_id}_{RGI_L_id[-10:]}_{tlim[0]}-{tlim[1]}_ticoi_{ticoi_kwargs['ticoi_name']}.png"
            ),
            **kwargs
        )
        tkn.plot_itslive_ticoi(
            ds_i=ds_i, ds_t=ds_t, d_era5=d, df_unrest=volcano.unrest, df_eruptions=volcano.eruptions,
            ylim=ylim,
            filepath=os.path.join(
                dir_figs,
                f"{GVP_id}_{RGI_G_id}_{RGI_L_id[-10:]}_{tlim[0]}-{tlim[1]}_{ticoi_kwargs['ticoi_name']}.png"
            ),
            **kwargs
        )


def ids2df():
    df = pd.DataFrame(columns=["GVP_id", "RGI_G_id", "RGI_L_id"])
    for GVP_id in ids.keys():
        volcano = Volcano(gvp_id=GVP_id)
        RGI_G_ids = ids[GVP_id].keys()
        if len(RGI_G_ids) == 0:
            RGI_G_ids = [rgi_id[-8:] for rgi_id in volcano.glaciers.rgi_id]

        for RGI_G_id in RGI_G_ids:
            if RGI_G_id not in ids[GVP_id].keys() or len(ids[GVP_id][RGI_G_id]) == 0:
                RGI_L_ids = [
                    rgi_id[-8:] for rgi_id in volcano.centrelines.loc[
                        (volcano.centrelines.rgi_g_id == f"{RGI_v}-G-{RGI_G_id}") & (volcano.centrelines.is_main == 1),
                        "rgi_id"
                    ]
                ]
            else:
                RGI_L_ids = ids[GVP_id][RGI_G_id]

            RGI_G_id = f"{RGI_v}-G-{RGI_G_id}"

            for RGI_L_id in RGI_L_ids:
                RGI_L_id = f"{RGI_v}-L-{RGI_L_id}"
                if len(volcano.centrelines.loc[
                           (volcano.centrelines.rgi_g_id == RGI_G_id) & (volcano.centrelines.rgi_id == RGI_L_id)
                       ]) == 0:
                    raise ValueError(f"No match for {GVP_id}, {RGI_G_id}, and {RGI_L_id}")
                if volcano.centrelines.loc[volcano.centrelines.rgi_id == RGI_L_id, "length_m"].values[0] < 2e3:
                    continue

                df1 = pd.DataFrame(data={"GVP_id": GVP_id, "RGI_G_id": RGI_G_id, "RGI_L_id": RGI_L_id}, index=[0])
                if len(df) == 0:
                    df = df1.copy()
                else:
                    df = pd.concat([df, df1], ignore_index=True)

    return df


def main():
    df_ids = ids2df()

    tasks = [
        mp.Process(
            target=run_volcano_glacier_centreline_df,
            kwargs={"df": df_ids.loc[[ind for ind in df_ids.index if ind % n_parallel == i]]}
        ) for i in range(min(len(df_ids), n_parallel))
    ]
    for task in tasks:
        task.start()

    for task in tasks:
        task.join()


if __name__ == "__main__":
    main()
