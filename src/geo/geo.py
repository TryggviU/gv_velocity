import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import rasterio


def bounding_box(gdf, padding=0):
    """
    Create a bounding box around a GeoDataFrame.

    :param gdf: A GeoDataFrame, possibly with multiple geometries.
    :param padding: Padding/buffer around the GeoDataFrame to be applied. Given as a fraction of width/height, so
        bbox_width/height = gdf_width/height * (1 + 2 * padding)

    :return: A "square" bounding box GeoDataFrame, with a buffer of padding*width/height.
    """

    # The extent of the GeoDataFrame
    if type(gdf) is gpd.GeoDataFrame:
        minx, miny, maxx, maxy = gdf.total_bounds
    elif type(gdf) is rasterio.io.DatasetReader:
        minx, miny, maxx, maxy = gdf.bounds
    else:
        raise TypeError("The input is of wrong type.")

    # Envelope the bounding box with small padding.
    dx = padding * (maxx - minx)
    dy = padding * (maxy - miny)

    # Create edge points.
    p1 = Point(minx - dx, miny - dy)
    p2 = Point(minx - dx, maxy + dy)
    p3 = Point(maxx + dx, maxy + dy)
    p4 = Point(maxx + dx, miny - dy)

    # Create edges.
    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    # Connect the edges to a "square" polygon.
    bb_polygon = Polygon([np1, np2, np3, np4])

    # Create a GeoDataFrame from the polygon.
    bbox = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=["geometry"], crs=gdf.crs)

    # Return the bounding box.
    return bbox.envelope


def points2bbox(minx, maxx, miny, maxy, crs):
    """
    Create a bounding box from x and y limits.

    :return: A "square" bounding box GeoDataFrame.
    """

    # Create edge points.
    p1 = Point(minx, miny)
    p2 = Point(minx, maxy)
    p3 = Point(maxx, maxy)
    p4 = Point(maxx, miny)

    # Create edges.
    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    # Connect the edges to a "square" polygon.
    bb_polygon = Polygon([np1, np2, np3, np4])

    # Create a GeoDataFrame from the polygon.
    bbox = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=["geometry"], crs=crs)

    # Return the bounding box.
    return bbox.envelope


def fix_antimeridian_pass(gdf, ref_point):
    s = np.sign(ref_point.geometry.values[0].xy[0][0])

    x = np.array(gdf.geometry.values[0].exterior.xy[0])
    y = np.array(gdf.geometry.values[0].exterior.xy[1])

    x[s*x < 0] += s*360

    geom = Polygon([[x[i], y[i]] for i in range(len(x))])

    return gpd.GeoDataFrame(gpd.GeoSeries(geom), columns=["geometry"], crs="EPSG:4326")


def create_point_buffer(point, rad):
    """
    Create a buffer surrounding a point.
    :param point: GeoDataFrame / GeoSeries
    :param rad: The radial distance that makes up the buffer in metres.
    :return: A GeoDataFrame of the circular buffer zone.
    """

    # First change projection to UTM.
    point = point.to_crs(point.estimate_utm_crs())

    # Create a reference point.
    if point.geom_type.values[0] == "Point":
        ref_point = point.to_crs("EPSG:4326")
    else:
        ref_point = point.centroid.to_crs("EPSG:4326")

    # Create the buffer.
    buffer = point.buffer(rad)

    # Convert back to WGS84.
    buffer = buffer.to_crs("EPSG:4326")

    # Fix the buffer if it passes the antimeridian.
    if not buffer.is_valid.values[0]:
        buffer = fix_antimeridian_pass(gdf=buffer, ref_point=ref_point)

    return buffer


def shapes_within_rad(point, shapes, rad):
    """
    Find all shapes/geometries that overlap with a radial buffer zone.

    :param point: GeoDataFrame/GeoSeries of a point.
    :param shapes: GeoDataFrame/GeoSeries of shapes/geometries.
    :param rad: The radial distance that makes up the buffer in metres.
    :return: GeoDataFrames of the shapes/geometries that fall within the buffer zone (cut and full).
    """
    # Create the buffer.
    buffer = create_point_buffer(point=point, rad=rad)

    # Clip the shapes with the buffer (portions of the shapes within the buffer).
    shapes_in_buffer_cut = shapes.clip(buffer, keep_geom_type=True)
    # Save the full shape geometries, i.e. any shape that reaches within the buffer.
    if shapes_in_buffer_cut.empty:
        shapes_in_buffer = shapes_in_buffer_cut
    else:
        shapes_in_buffer = shapes.loc[[s for s in shapes_in_buffer_cut.index]]

    return shapes_in_buffer_cut, shapes_in_buffer


def shapes_within_gdf(gdf, shapes):
    """
    Find all shapes/geometries that overlap with a GeoDataFrame.

    :param gdf: GeoDataFrame a geometry.
    :param shapes: GeoDataFrame/GeoSeries of shapes/geometries.
    :return: GeoDataFrames of the shapes/geometries that fall within the GeoDataFrame (cut and full).
    """

    # Clip the shapes with the buffer (portions of the shapes within the buffer).
    shapes_in_gpd_cut = shapes.clip(gdf, keep_geom_type=True)
    # Save the full shape geometries, i.e. any shape that reaches within the buffer.
    if shapes_in_gpd_cut.empty:
        shapes_in_gpd = shapes_in_gpd_cut
    else:
        shapes_in_gpd = shapes.loc[[s for s in shapes_in_gpd_cut.index]]

    return shapes_in_gpd_cut, shapes_in_gpd


def compute_area(gdf):
    # Change projection to UTM.
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    return gdf.area.sum()


def compute_distance(f1, f2, f3=gpd.GeoDataFrame()):
    """
    A function that computes the shortest distance between a specific feature (f1) and another feature(s) (f2).
    An optional argument of other features (f3) that might be closer to f1 than f2 is provided. If f3 are closer to f2
    than f1 is to f2, the function returns the distance as NaN.

    :param f1: A GeoDataFrame consisting of a specific feature.
    :param f2: A GeoDataFrame consisting of features to have their distance from f1 computed.
    :param f3: (Optional) A GeoDataFrame with features that might be closer to f1.

    :return: The f2 GeoDataFrame with an additional "distance" column.
    """

    # Add a distance column to the f2 GeoDataFrame.
    f2["distance"] = np.zeros(len(f2))

    # Ensure that all GeoDataFrames are in the same UTM CRS.
    f2 = f2.to_crs(f2.estimate_utm_crs())
    f1 = f1.to_crs(f2.crs)
    f3 = f3.to_crs(f2.crs)

    # Iterate through all features in the f2 GeoDataFrame.
    for i in range(len(f2)):
        # Create a GeoSeries object of i-th feature within f2.
        f2i = gpd.GeoSeries(f2.loc[[i]].geometry.to_list(), crs=f2.crs)

        # Compute the shortest line between f1 and the i-th f2 feature.
        sl = f1.shortest_line(f2i)
        x = sl.get_coordinates().x.values
        y = sl.get_coordinates().y.values

        # Add the length of the shortest line to the "distance" column of the f2 features.
        f2.loc[i, "distance"] = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)

        # Exclude the distance if f3 features are closer to f2.
        if len(f3) > 0:
            for j in range(len(f3)):
                f3j = gpd.GeoSeries(f3.iloc[[j]].geometry.to_list(), crs=f2.crs)

                # Compute the shortest line between the j-th f2 feature and the i-th f2 feature.
                sl = f3j.shortest_line(f2i)
                x = sl.get_coordinates().x.values
                y = sl.get_coordinates().y.values

                if np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) < f2.loc[i, "distance"]:
                    f2.loc[i, "distance"] = np.nan

    return f2


def set_scalebar(E, N):
    """
    Set the scalebar size.

    :param E: East coordinate
    :param N: North coordinate
    :return: a distance element for the scalebar.
    """

    pts = gpd.GeoSeries(
        [Point(E, N), Point(E + 1, N)],
        crs="EPSG:4326"
    )
    pts = pts.to_crs(pts.estimate_utm_crs())

    return pts[0].distance(pts[1])


def simplify_polylines(gdf: gpd.GeoDataFrame, max_segment_length=200):
    """
    Simplify centre-line geometries of glaciers.

    :param polylines: Geodataframe of polyline(s) to be simplified.
    :param max_segment_length: The maximum segment length of the simplified glacier centre lines in metres.
    """
    crs = gdf.crs
    spl = gdf.to_crs(gdf.estimate_utm_crs())
    spl.geometry = spl.simplify(tolerance=max_segment_length/10).segmentize(max_segment_length=max_segment_length)

    return spl.to_crs(crs)  # Transform back to original coordinate system.


def profile_distance(polyline):
    points = polyline.to_crs(polyline.estimate_utm_crs()).get_coordinates().values

    distance = np.zeros(len(points))

    for i in range(1, len(distance)):
        distance[i] = distance[i-1] + np.sqrt((points[i, 0] - points[i-1, 0])**2 + (points[i, 1] - points[i-1, 1])**2)

    return np.max(distance) - distance  # distance[::-1]


def coords2points(coords):
    return gpd.GeoDataFrame(
        {"geometry": [Point(coords[i]) for i in range(len(coords))]},
        crs="EPSG:4326"
    )


def coords2polyline(coords):
    return gpd.GeoDataFrame(
        {"geometry": [LineString(coords)]},
        crs="EPSG:4326"
    )
