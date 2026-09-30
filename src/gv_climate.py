import os
import pandas as pd
# Modules
import tools.tools as tools
import era.era as era

# Set the root directory as the project directory.
dir_root = tools.set_root(path=os.getcwd(), name="gv_velocity")
# Data directories.
dir_data = os.path.join(dir_root, "data")
dir_data_proc = os.path.join(dir_root, "data_processed")
dir_era5 = tools.mkdir_ifnot_exist(
    path=os.path.join(dir_data, "ERA5"),
    path_proj=dir_root
)

# The RGI version code/phrase (e.g. RGI2000-v7.0).
RGI_v = "RGI2000-v7.0"

# Read in the volcanoes.
volcanoes = pd.read_csv(
    tools.find_files_within_path(
        path=dir_data,
        filename="GVP_Volcano_List_Holocene.csv"
    )[0], header=1, encoding='latin-1'
)


def era_volcano_path(GVP_id):
    # Name the destination
    return os.path.join(dir_era5, f"era5_{GVP_id}.nc")


def download_volcano_climate(GVP_id):
    # Download the ERA5 timeseries if they haven't been downloaded already.
    if not os.path.exists(era_volcano_path(GVP_id)):
        volcano = volcanoes.loc[volcanoes["Volcano Number"] == GVP_id]
        era.download_era5_timeseries(
            lat=volcano.Latitude.values[0],
            lon=volcano.Longitude.values[0],
            path=era_volcano_path(GVP_id)
        )


def main():
    for GVP_id in volcanoes["Volcano Number"].values:
        print(GVP_id)
        download_volcano_climate(GVP_id=GVP_id)


if __name__ == "__main__":
    main()
