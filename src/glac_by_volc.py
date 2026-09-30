"""
Script used to locate glaciers within a radial distance of a volcano.

Input: Data files from RGI - http://www.glims.org/rgi_user_guide/welcome.html
       Data file* from GVP - https://volcano.si.edu/volcanolist_holocene.cfm
       Data files provided by 'volc_rgi_regions.py' output.
Output: .csv and shape files of glaciers within a radial distance of a volcano.

*file must be converted from .xml to .csv prior to usage.
"""

import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import argparse
# Modules.
import geo.geo as geo
import gvf.gvf as gv
import tools.tools as tools

# Set the root directory as the project directory.
dir_root = tools.set_root(path=os.getcwd(), name="gv_velocity")
# Data directories.
dir_data = os.path.join(dir_root, "data")
dir_data_proc = os.path.join(dir_root, "data_processed")

# The RGI version code/phrase (e.g. RGI2000-v7.0).
RGI_v = os.path.basename(
    tools.find_files_within_path(
        path=os.path.join(dir_data, "RGI"),
        filename="RGI"
    )[0]
)[0:12]

# Read in the RGI region codes/names.
RGI_regions = gv.read_rgi(
    path=tools.find_files_within_path(
        path=dir_data,
        filename="o1regions-summary.csv"
    )[0]
)["long_code"].drop_duplicates().tolist()

# Initiate the argument parser.
parser = argparse.ArgumentParser()
# Add arguments.
parser.add_argument("-i", "--RGI_ids", action="store", nargs="*", default=RGI_regions,
                    help="Select RGI regional ID code(s).")
parser.add_argument("-r", "--radius", action="store", default=10, type=float,
                    help="Maximum radial distance, in kilometres, of search buffer for glaciers surrounding volcano.")
parser.add_argument("-d", "--display", action="store", type=bool, default=False, choices=[False, True],
                    help="Plot figures of each volcano, buffer and glaciers within buffer.")
# Read arguments from the command line
args = parser.parse_args()
# Change from kilometres to metres.
args.radius = args.radius * 1e3
print(args)


def find_centrelines(glaciers, rgi_centrelines):
    gdf = gpd.GeoDataFrame()

    for i in glaciers.index:
        gdf = tools.join_df(
            df1=gdf,
            df2=rgi_centrelines.loc[rgi_centrelines["rgi_g_id"] == glaciers.loc[i, "rgi_id"]]
        )

    return gdf


def save_to_shp_csv(volcano, glaciers, centrelines, glacier_coverage, GVP_id, RGI_id, rad=args.radius):
    # Create the subdir.
    volc_subdir = tools.mkdir_ifnot_exist(
        path=os.path.join(dir_data_proc,
                          "regional_files",
                          f"{RGI_v}-GV",
                          f"{RGI_v}-GV-{RGI_id}",
                          f"{RGI_id}_{round(rad / 1e3 + 1e-6, 1)}km",
                          f"{GVP_id}"),
        path_proj=dir_root
    )

    # Save the volcano info to an attribute .csv file.
    df = pd.DataFrame(volcano.drop(columns="geometry"))
    df["N Glaciers"] = len(glaciers)  # Add the number of glaciers to the attribute table.
    df["Glacier Coverage"] = glacier_coverage  # Add the glacier coverage to the attribute table.
    df.to_csv(
        os.path.join(
            volc_subdir,
            f"{GVP_id}_{round(rad / 1e3 + 1e-6, 1)}km-attributes.csv"),
        index=False
    )

    # Save a shape file of the glaciers.
    glaciers.to_file(
        os.path.join(
            volc_subdir,
            f"{GVP_id}_{round(rad / 1e3 + 1e-6, 1)}km-glaciers.shp"
        )
    )

    # Save a shape file of the glacier centre lines.
    centrelines.to_file(
        os.path.join(
            volc_subdir,
            f"{GVP_id}_{float(rad/1e3)}km-centrelines.shp"
        )
    )


def plot_volc_glac(volcano, glaciers, glac_coverage, name, rad=args.radius):
    # An intermediary figure to get the color for the legend of figure.
    plt.figure(0)
    color_proxy = plt.plot(
        [0, 1],
        [0, 1],
        color="tab:blue",
        alpha=0.75,
        linestyle="",
        marker="s",
        label="Ice coverage: {0}%".format(round(100 * glac_coverage, 1))
    )
    plt.clf()
    plt.close(0)

    # The actual figure.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title(name)
    glaciers.plot(
        ax=ax,
        color="tab:blue",
        alpha=0.75
    )
    geo.create_point_buffer(volcano, rad).plot(ax=ax, color="tab:purple", alpha=0.25)
    volcano.plot(ax=ax, color="tab:purple")
    ax.legend(handles=color_proxy, loc='lower right')
    plt.show()


def find_glac_within_rad(RGI_id, rad=args.radius, display=args.display):
    """
    Iterate through the volcanoes within a RGI region, and find glaciers within a specific radial distance.

    :param RGI_id:  RGI regional codes - See codes in: RGI2000-v7.0-o1regions-summary.csv
    :param rad: The radial distance (<100 km) that glaciers may be from a volcano.
    :param display: Sets whether each volcano and associated glaciers is plotted or not.
    :return:
    """

    print(f"RGI region: {RGI_id}")

    # Check if regional subdirectory exists.
    volc_subdir = os.path.join(dir_data_proc, "regional_files", f"{RGI_v}-V", f"{RGI_v}-V-{RGI_id}")
    if not os.path.exists(volc_subdir):
        print(" - No subdirectory found for: {0}".format(RGI_id))
        print("   - Either no volcanoes within RGI region, or 'volc_in_rgi.py' did not run correctly.")
        return

    # Read in the regional volcanoes.
    gvp_regional = gpd.read_file(
        tools.find_files_within_path(
            path=dir_data_proc,
            filename=f"{RGI_v}-V-{RGI_id}.shp"
        )[0]
    )
    # Read in the regional glaciers.
    rgi_regional = gpd.read_file(
        tools.find_files_within_path(
            path=dir_data,
            filename=f"{RGI_v}-G-{RGI_id}.shp"
        )[0]
    )
    # Read in the regional glacier centre lines.
    rgi_centrelines = gpd.read_file(
        tools.find_files_within_path(
            path=dir_data,
            filename=f"{RGI_v}-L-{RGI_id}.shp"
        )[0]
    )

    # Iterate through all the regional volcanoes. *ESRI shape files allow for max 10 characters in column name.
    for i, GVP_id in enumerate(gvp_regional["Volcano Number"[0:10]]):
        # Extract a volcano from the regional dataset.
        volcano = gvp_regional[i:i + 1]
        print(" - Volcano: {0}".format(volcano["Volcano Name"[0:10]].values[0]),
              "- # {0}".format(volcano["Volcano Number"[0:10]].values[0]))

        # Find all the glacier geometries that overlap the buffer.
        glaciers_cut, glaciers = geo.shapes_within_rad(
            point=volcano,
            shapes=rgi_regional,
            rad=rad
        )

        # Compute the relative glacier coverage.
        if glaciers_cut.empty:
            glac_coverage = 0
        else:
            glac_coverage = geo.compute_area(gdf=glaciers_cut) / (math.pi * rad**2)

            # Find the glacier centrelines.
            centrelines = find_centrelines(glaciers=glaciers, rgi_centrelines=rgi_centrelines)

            # Save the results to .shp and .csv files.
            save_to_shp_csv(
                volcano=volcano,
                glaciers=glaciers,
                centrelines=centrelines,
                glacier_coverage=glac_coverage,
                RGI_id=RGI_id,
                GVP_id=GVP_id
            )

            if display:
                plot_volc_glac(
                    volcano=volcano,
                    glaciers=glaciers,
                    glac_coverage=glac_coverage,
                    name="{0}: {1}".format(RGI_id, volcano["Volcano Name"[0:10]].values[0])
                )

        print("   - Glacier coverage: {0}".format(glac_coverage))


def main():
    # Check if input variables are valid and clean up.
    if isinstance(args.RGI_ids, list):
        # Check if there are duplicates within list.
        if len(args.RGI_ids) != len(set(args.RGI_ids)):
            raise ValueError("Duplicates detected within regional_codes.")
        else:
            # Check if the regional codes are valid.
            if any(rc not in RGI_regions for rc in args.RGI_ids):
                raise ValueError("Some regional code is not a valid RGI code.")

            # Drop Antarctic Mainland from the RGI list.
            if "20_antarctic_mainland" in args.RGI_ids:
                print("Dropping RGI region '20_antarctic_mainland'.")
                args.RGI_ids.remove("20_antarctic_mainland")

        # Set the limit to the radial distance as 100 km.
        if args.radius > 100e3:
            raise ValueError("The radial distance must be smaller than 100 km.")
    else:
        raise TypeError("regional_codes must be a list of RGI regional codes.")

    # Find glaciers for all volcanoes in all RGI regions.
    for RGI_id in args.RGI_ids:
        find_glac_within_rad(RGI_id=RGI_id)


if __name__ == "__main__":
    main()
