import os
import pandas as pd


def read_rgi(path):
    # Read in the RGI attributes.
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError("RGI attributes file not recognised: {file}".format(file=path))


def read_gvp(path):
    # Read in the GVP global attributes.
    if os.path.exists(path):
        return pd.read_csv(path, header=1, encoding='latin-1')
    else:
        raise FileNotFoundError("GVP global attributes file not recognised: {file}.".format(file=path))


def gvp_eruptions(xlsx, GVP_id):
    """
    Find all documented eruptions of a given volcano from the Smithsonian Global Volcanism Program's eruption list.
    :param xlsx: Path to data file from GVP - https://volcano.si.edu/search_eruption.cfm
    :param GVP_id: The volcano ID number from the Smithsonian Global Volcanism Program.

    :return : Pandas dataframe of all eruptions of the input volcano.
    """
    df = pd.read_excel(
        os.path.join(xlsx),
        sheet_name="Eruption List",
        header=1
    )

    return df.loc[df["Volcano Number"] == GVP_id]


def gvp_format_dates(df):
    """
    Format the start and end dates in the Smithsonian's Global Volcanism Program database files to a readable format.
    Change Start/End Year, Month, Day to a DateTime and Start/End Uncertainty to TimeDelta.

    :param df: The GVP database as a Pandas DataFrame
    :return :  The input DataFrame with added columns 'DateStart', 'DeltaStart', 'DateEnd', and 'DeltaEnd'.
    """

    if len(df) == 0:
        return df

    df[[
        "Start Year Uncertainty", "Start Month", "Start Day", "Start Day Uncertainty",
        "End Year Uncertainty", "End Month", "End Day", "End Day Uncertainty"
    ]] = df[[
        "Start Year Uncertainty", "Start Month", "Start Day", "Start Day Uncertainty",
        "End Year Uncertainty", "End Month", "End Day", "End Day Uncertainty"
    ]].fillna(value=0)

    df = df.loc[df["Start Year"] >= 1900, :]  # Only consider 20th century activity and onwards.

    # Initialise the DateTime and TimeDelta columns.
    df.insert(len(df.columns), "DateStart", pd.to_datetime("20011002"))
    df.insert(
        len(df.columns), "DeltaStart",
        pd.to_timedelta(df["Start Day Uncertainty"], "D") + pd.to_timedelta(df["Start Year Uncertainty"] * 365.25, "D"),
    )
    df.insert(len(df.columns), "DateEnd", pd.to_datetime("20011002"))
    df.insert(
        len(df.columns), "DeltaEnd",
        pd.to_timedelta(df["End Day Uncertainty"], "D") + pd.to_timedelta(df["End Year Uncertainty"] * 365.25, "D")
    )

    for s in ["Start", "End"]:
        df.loc[
            (df[f"{s} Month"] == 0) & (df[f"Delta{s}"] == pd.to_timedelta(0, "D")),
            f"Delta{s}"
        ] = pd.to_timedelta(365.25/2, "D")  # If no M and D specified: set Dt as half a year.
        df.loc[
            (df[f"{s} Month"] == 0) & (df[f"{s} Day"] == 0),
            [f"{s} Month", f"{s} Day"]
        ] = [7, 1]  # If no M and D specified: set M=7 and D=1 (middle of year)
        df.loc[
            df[f"{s} Day"] == 0, f"Delta{s}"
        ] = pd.to_timedelta(14, "D")  # If no D specified: set Dt as two weeks.
        df.loc[
            df[f"{s} Day"] == 0, f"{s} Day"
        ] = 15  # If no D specified: set D=15 (middle of month)
        df.loc[:, f"Date{s}"] = pd.to_datetime(dict(year=df[f"{s} Year"], month=df[f"{s} Month"], day=df[f"{s} Day"]))

    df.loc[df[f"End Year"].isna(), ["DateEnd", "DeltaEnd"]] = df.loc[
        df[f"End Year"].isna(), ["DateStart", "DeltaStart"]
    ].values

    return df


def gvp_unrest_eruptions(dir_csv, GVP_id):
    try:
        df_unrest = pd.read_csv(
            os.path.join(dir_csv, f"{GVP_id}-unrest.csv")
        )
    except FileNotFoundError:
        df_unrest = pd.DataFrame()

    try:
        df_eruptions = pd.read_csv(
            os.path.join(dir_csv, f"{GVP_id}-eruptions.csv")
        )
    except FileNotFoundError:
        df_eruptions = pd.DataFrame()

    return gvp_format_dates(df=df_unrest), gvp_format_dates(df=df_eruptions)