import os
import numpy as np
import pandas as pd
import earthkit.data as ekd
import xarray as xr
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL


def download_invariants(dir_era5, force=False):
    """
    Download the invariant fields from the ERA5 datasets: land-sea-mask and elevation.
    """
    datasets = ["reanalysis-era5-single-levels", "reanalysis-era5-land"]

    for dataset in datasets:
        file = os.path.join(dir_era5, f"{dataset}.grib")

        print(f"Downloading ERA5 dataset: {dataset}")

        if not force and os.path.exists(file):
            print(" - Skipping, as dataset already downloaded.")
            continue

        # Download the data from the Copernicus Climate Data Store (CDS).
        invariants = ekd.from_source(
            'cds',
            dataset,
            {
                'product_type': 'reanalysis',
                'variable': ['land_sea_mask', 'geopotential'],
                'year': "2025",
                'month': "03",
                'day': "30",
                'time': '12:00',
                'data_format': 'grib',
            },
        )
        with open(file, "wb") as fp:
            invariants.to_target("file", fp)


def find_nearest_climate_point(lon, lat, dir_era5):
    """
    Checking if there is a point in the ERA5-land dataset that can be used to represent the coordinates.
    ERA5-land : 0.10 x 0.10 degree resolution
    ERA5      : 0.25 x 0.25 degree resolution
    """

    # Read in the ERA5-land dataset and convert to xarray.
    era5_land = os.path.join(dir_era5, "reanalysis-era5-land.grib")
    ds = ekd.from_source("file", era5_land)
    xa = ds.to_xarray()
    # Load the ERA5-land land-sea-mask around the coordinates for an area the ~size of ERA5 cell (0.25x025 degrees).
    lsm = xa.lsm.sel(
        longitude=np.arange(lon-0.2, lon+0.21, 0.1)+360,
        latitude=np.arange(lat-0.2, lat+0.21, 0.1),
        method='nearest'
    ).values

    # Return the coordinates if ERA5-land has no land points.
    if (lsm < 0.5).all():
        return {"lat": lat, "lon": lon, "era": "ERA5"}
    # Find the nearest land point to the input coordinates in the ERA5-land dataset.
    else:
        X, Y = np.meshgrid(np.arange(-0.2, 0.21, 0.1), np.arange(-0.2, 0.21, 0.1))
        dist = np.round(np.sqrt(X**2 + Y**2), 6)  # The distance between points in ERA5-land.

        for d in np.unique(dist):  # Iterate through points from shortest to longest distance from input coordinates.
            if np.max(lsm[dist == d]) > 0.5:  # Check if there are land points.
                # Select the coordinates with:
                #   the shortest distance from input coordinates;
                #   and the first highest land value.
                lon += X[dist == d][lsm[dist == d] == np.max(lsm[dist == d])][0]
                lat += Y[dist == d][lsm[dist == d] == np.max(lsm[dist == d])][0]
                break

        return {"lat": lat, "lon": lon, "era": "ERA5-land"}


def download_era5_timeseries(lat, lon, path, date_start="1940-01-01", date_end="2025-04-30"):
    """
    Dataset: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries
    """
    ds = ekd.from_source(
        "cds",
        "reanalysis-era5-single-levels-timeseries",
        {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature', 'total_precipitation'],
            "date": [date_start, date_end],
            'data_format': 'netcdf',
            'location': {"longitude": lon, "latitude": lat}
        }
    )

    xa = ds.to_xarray()
    xa.to_netcdf(path)


def read_era5_timeseries(path):
    ds = xr.open_dataset(path, engine="netcdf4")
    return ds.to_pandas()


def compute_daily_stats(df):
    """
    Compute daily mean temperatures and daily total precipitation.

    :param df: A pandas dataframe containing ERA5 data from read_era5_timeseries.
    """
    df["DateTime"] = pd.to_datetime(df.index)
    df_daily = df.loc[:, ["DateTime", "t2m", "tp"]].groupby(pd.Grouper(key="DateTime", freq='1D')).mean()
    df_daily.loc[:, "tp"] *= 24  # Change precipitation from average per day to total.
    return df_daily


def compute_temp_stats(df, freq):
    """
    Compute temporal mean temperatures and total precipitation.

    :param df: A pandas dataframe containing ERA5 data from read_era5_timeseries.
    :param freq: The temporal frequency to compute means and totals - '1D', '1M', '1Y'
    """
    df["DateTime"] = pd.to_datetime(df.index)
    df_temp = df.loc[:, ["DateTime", "t2m", "tp"]].groupby(pd.Grouper(key="DateTime", freq=freq)).mean()
    df_temp.loc[:, "tp"] *= 24  # Change precipitation from average per day to total.
    return df_temp


def normalised_nonseasonal_timeseries(ts, freq, period, method=None):
    def normalised_nonseasonal(signal, seasonal):
        nonseasonal = signal - seasonal
        nonseasonal -= nonseasonal.mean()
        nonseasonal /= max(nonseasonal.max(), -nonseasonal.min())
        return nonseasonal

    # Run the seasonal decomposition.
    if freq == "1YS":  # No seasonal component for yearly data.
        res = {key: ts[key].mean() for key in ts.keys()}

        for key in ts.keys():
            ts[key + "_ns"] = normalised_nonseasonal(ts[key], res[key])
    else:
        if method == "STL":
            res = {key: STL(ts[key].dropna(), period=period).fit() for key in ts.keys()}
        elif method == "MSTL":
            res = {key: MSTL(ts[key].dropna(), periods=period).fit() for key in ts.keys()}
        else:
            res = {key: seasonal_decompose(ts[key].dropna(), period=period) for key in ts.keys()}

        for key in ts.keys():
            ts[key + "_ns"] = normalised_nonseasonal(ts[key], res[key].seasonal)

    return ts, res


def read_and_process_ERA5(filepath):
    df = read_era5_timeseries(path=filepath)
    df["t2m"] -= 273.15  # Change temperature from K to C.
    df["tp"] *= 1000  # Change precipitation total from m to mm.

    d = {}
    for freq, period in zip(["1D", "1MS", "1YS"], [365, 12, 1]):
        d[f"ts_{freq}"], d[f"res_{freq}"] = normalised_nonseasonal_timeseries(
            ts=compute_temp_stats(df=df, freq=freq),
            freq=freq,
            period=period
        )

    return d
