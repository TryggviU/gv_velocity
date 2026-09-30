"""
Script used to locate volcanoes within RGI regions.
Input: Data files from RGI - http://www.glims.org/rgi_user_guide/welcome.html
       Data file* from GVP - https://volcano.si.edu/volcanolist_holocene.cfm
Output: .csv and shape files of volcanoes within each RGI region.

*file must be converted from .xml to .csv prior to usage.
"""

import os
import pandas as pd
import geopandas as gpd
# Modules.
import gvf.gvf as gv
import tools.tools as tools

# Set the root directory as the project directory.
dir_root = tools.set_root(path=os.getcwd(), name="gv_velocity")
# Data directories.
dir_data = os.path.join(dir_root, "data")
dir_data_proc = os.path.join(dir_root, "data_processed")

# The RGI version code/phrase (e.g. RGI2000-v7.0).
RGI_version = os.path.basename(
    tools.find_files_within_path(
        path=os.path.join(dir_data, "RGI"),
        filename="RGI"
    )[0]
)[0:12]


def read_rgi_gvp():
    """
    Read in the RGI regions and GVP volcanoes.

    :return: RGI regions and GVP volcanoes as GeoDataFrames
    """

    # Read in shape file of the RGI regions as a GeoDataFrame.
    rgi = gpd.read_file(
        tools.find_files_within_path(path=dir_data,
                                     filename="o1regions.shp")[0]
    )

    # Read in the GVP volcanoes as a GeoDataFrame.
    gvp_attributes = gv.read_gvp(
        tools.find_files_within_path(
            path=dir_data,
            filename="GVP_Volcano_List_Holocene.csv")[0]
    )
    gvp = gpd.GeoDataFrame(
        gvp_attributes,
        geometry=gpd.points_from_xy(gvp_attributes.Longitude, gvp_attributes.Latitude),
        crs="EPSG:4326"
    )

    return rgi, gvp


def result_dir_setup():
    """
    Set up the result directory structure:
    data_processed
        global_files
        |   RGI2000-v7.0-GV
        |   RGI2000-v7.0-V
        regional_files
        |   RGI2000-v7.0-GV
        |   RGI2000-v7.0-V
    """
    # Create the GVP global and regional subdirectories.
    for subdir in ["global_files", "regional_files"]:
        for subsubdir in [f"{RGI_version}-{char}" for char in ["V", "GV"]]:
            tools.mkdir_ifnot_exist(
                os.path.join(dir_data_proc,
                             subdir,
                             subsubdir),
                dir_root
            )


def save_gvp2shp_csv(gvp, dir_gvp, RGI_id):
    """
    Save the GVP regional GeoDataFrame to a .shp and .csv file.

    :param gvp: The GVP regional GeoDataFrame
    :param dir_gvp: The result directory.
    :param RGI_id: The RGI regional code.
    """

    # Save the shape file.
    gvp.to_file(os.path.join(
        dir_gvp,
        f"{RGI_version}-V-{RGI_id}.shp")
    )
    # Save a csv file with the data.
    gvp.drop(columns="geometry").to_csv(
        os.path.join(dir_gvp,
                     f"{RGI_version}-V-{RGI_id}-attributes.csv"),
        index=False
    )


def get_rgi_region(rgi, RGI_id):
    """
    Get a GeoDataFrame of a single RGI region from the collection of regions.

    :param rgi: A GeoDataFrame of all RGI regions.
    :param RGI_id: A RGI regional code.
    :return: A GeoDataFrame of single RGI region.
    """

    # Find the index/indices of the region code within the RGI GeoDataFrame.
    inds = rgi["long_code"][rgi["long_code"] == RGI_id].index

    # Save the specific RGI region.
    rgi_regional = rgi[inds[0]:inds[0] + 1]
    # Append the RGI region if needed.
    if len(inds) > 1:
        for ind in inds[1:]:
            rgi_regional = pd.concat([rgi_regional, rgi[ind:ind + 1]])

    return rgi_regional


def find_volc_in_rgi():
    """
    Find all volcanoes within the RGI regions.
    """

    # Set up the result directories.
    result_dir_setup()

    # Read in the RGI regions and GVP volcanoes as GeoDataFrames.
    rgi, gvp = read_rgi_gvp()

    # Read in the RGI region codes/names.
    RGI_ids = gv.read_rgi(
        path=tools.find_files_within_path(
            path=dir_data,
            filename="o1regions-summary.csv"
        )[0]
    )

    # Iterate through the RGI regions (excluding duplicates).
    for RGI_id in RGI_ids["long_code"].drop_duplicates():
        # Get a GeoDataFrame of the RGI region.
        rgi_regional = get_rgi_region(rgi=rgi, RGI_id=RGI_id)

        # Clip the GVP GeoDataFrame with the RGI region.
        gvp_regional = gvp.clip(rgi_regional)

        # Save the results if any.
        if not gvp_regional.empty:
            # Create the subdir.
            dir_regional_results_v = tools.mkdir_ifnot_exist(
                os.path.join(dir_data_proc,
                             "regional_files",
                             f"{RGI_version}-V",
                             f"{RGI_version}-V-{RGI_id}"),
                dir_root
            )

            # Save the results.
            save_gvp2shp_csv(
                gvp=gvp_regional,
                dir_gvp=dir_regional_results_v,
                RGI_id=RGI_id
            )


def main():
    # Execute the finding of volcanoes within specific RGI regions.
    find_volc_in_rgi()


if __name__ == "__main__":
    main()
