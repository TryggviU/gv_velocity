import os
import numpy as np
import pandas as pd


def set_root(path, name):
    # Check that the filename is included within the path.
    if name not in path:
        raise NotADirectoryError(f"Directory {name} is not a part of path: {path}.")

    # Set the working directory as the project directory.
    if os.path.split(path)[1] == name:
        return path
    else:
        return set_root(path=os.path.split(path)[0], name=name)


def find_files_within_path(path, filename):
    """
    Find files containing a specific filename within a specified path.

    :param path: A path that should contain the filename.
    :param filename: The (portion of a) filename.
    :return: A list of paths to files containing the filename.
    """

    # Make sure that the input path exists.
    if not os.path.exists(path=path):
        raise NotADirectoryError("The path: {0} does not exist".format(path))

    # Find all instances of the filename within the path.
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if filename in file:
                file_list.append(os.path.join(root, file))

    # Raise an error if the file does not exist.
    if not file_list:
        raise FileNotFoundError(f"No files including the filename: {filename} could be found in path: {path}")

    return file_list


def find_subdirs_within_path(path, dirname):
    """

    :param path:
    :param filename:
    :return:
    """

    # Make sure that the input path exists.
    if not os.path.exists(path=path):
        raise NotADirectoryError("The path: {0} does not exist".format(path))

    # Find all instances of the filename within the path.
    subdir_list = []
    for root, dirs, files in os.walk(path):
        for subdir in dirs:
            if dirname in subdir:
                subdir_list.append(os.path.join(root, subdir))

    # Raise an error if the file does not exist.
    if not subdir_list:
        raise FileNotFoundError(f"No subdirectories including the name: {dirname} could be found in path: {path}")

    return subdir_list


def mkdir_ifnot_exist(path, path_proj):
    if os.path.exists(path):
        return path
    else:
        # Make sure that the path is within the project directory scope.
        if path_proj not in path:
            raise ValueError(f"The specified path is outside the project directory scope.\n"
                             f"Project path: {path_proj}\n"
                             f" Chosen path: {path}")

        # If the parent directory exists, then create the subdirectory.
        if os.path.exists(os.path.dirname(path)):
            if not os.path.exists(path):
                os.mkdir(path)

        # Else create the parent directory(ies) of the path recursively.
        else:
            mkdir_ifnot_exist(os.path.dirname(path), path_proj)
            mkdir_ifnot_exist(path, path_proj)

        return path


def clear_temp_files(path):
    """
    !!! BE CAREFUL !!!
    Deletes all files from a directory.
    Restoration of deleted files is NOT possible.

    :param path: Path to a temporary directory, which must be named 'temp', where all files should be deleted.
    :return:
    """

    # Path must exist and be named "temp"
    if os.path.exists(path) and "temp" in path:
        # List of all subdirectories and files.
        sub_items = os.listdir(path=path)
        for item in sub_items:
            # If the item is a directory, delete its contents first.
            if os.path.isdir(os.path.join(path, item)):
                clear_temp_files(path=os.path.join(path, item))
                os.rmdir(path=os.path.join(path, item))
            else:
                os.remove(path=os.path.join(path, item))
    else:
        raise NameError("Some portion of the input directory must be named 'temp'.")


def join_df(df1, df2):
    """
    Join two DataFrames (works also for GeoDataFrames). The two DataFrames must have the same columns.

    :param df1: The first DataFrame.
    :param df2: The second DataFrame.
    :return: The combined DataFrame.
    """

    def check_dfs():
        if len(df1.columns) != len(df2.columns):
            print("df1", df1.columns)
            print("df2", df2.columns)
            raise ValueError("The columns of the two DataFrames do not match.")
        elif set(df1.columns) != set(df2.columns):
            print("df1", df1.columns)
            print("df2", df2.columns)
            raise ValueError("The columns of the two DataFrames do not match.")

    if len(df1) == 0:
        return df2
    elif len(df2) == 0:
        return df1
    else:
        check_dfs()
        return pd.concat([df1, df2], ignore_index=True)


def df_moving_average(data, x, y, xmin, xmax, dx):
    df = pd.DataFrame(
        {x: [xmin + i*dx for i in range(int((xmax-xmin)/dx + 1))],
         y: [float("nan") for i in range(int((xmax-xmin)/dx + 1))]}
    )

    for i in range(1, len(df[x])-1):
        df.loc[i, y] = data.loc[(df.loc[i - 1, x] < data[x]) & (data[x] < df.loc[i + 1, x]), y].mean()

    df.loc[0, y] = data.loc[data[x] < df.loc[1, x], y].mean()
    df.loc[int((xmax-xmin)/dx + 1), y] = data.loc[df.loc[int((xmax-xmin)/dx), x] < data[x], y].mean()

    return df


def df_shift(df, t, var, tvar):
    """
    Shift data in df to time series t.

    :param df: The DataFrame containing data in columns, and a DateTime column named 'tvar'.
    :param t: The time series, e.g., t = pd.date_range(start, stop).
    :param var: The name column to be shifted to t.
    :param tvar: The name of the DateTime column in df.

    :return : A dataframe consisting of resampled column df.var for time series t.
    """

    df_t = pd.DataFrame(data={var: np.nan}, index=pd.to_datetime(t))
    df_var = pd.DataFrame(data={var: df[var].values}, index=pd.to_datetime(df[tvar].values))
    df_t["t"] = df_t.index
    df_var["t"] = df_var.index

    # return pd.merge_ordered(
    #     df_t, df_var, on=["t", var]
    # ).set_index("t").interpolate(method="index", limit_direction="both").loc[df_t.index]

    df_f = pd.merge_ordered(
        df_t, df_var, on=["t", var]
    ).set_index("t").interpolate(method="index", limit_direction="both").loc[df_t.index]

    return df_f.loc[~df_f.index.duplicated(keep="first"), :]

