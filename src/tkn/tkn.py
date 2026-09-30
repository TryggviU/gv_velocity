import pandas as pd
import ultraplot as uplt
from matplotlib.pyplot import Line2D
import numpy as np


def plot_era5(axT, axP, d: dict, tlim=("2015-01-01", "2025-01-01"), legend=True, **kwargs):
    for key in d.keys():
        if key[:2] == "ts":
            d[key] = d[key].loc[(d[key].index >= tlim[0]) & (d[key].index <= tlim[1])]

    for ax, var, cmap, labels in zip(
            [axT, axP], ["t2m_ns", "tp_ns"], ["coolwarm", "PuOr"], (["Cold", "Mean", "Hot"], ["Dry", "Mean", "Wet"])
    ):
        ax.barh(d[f"ts_1MS"][var].index, [1], color=uplt.Colormap(cmap)((d[f"ts_1MS"][var].values + 1)/2))
        ax.barh(d[f"ts_1MS"][var].index, [-1], color=uplt.Colormap(cmap)((d[f"ts_1MS"][var].values + 1)/2))

        ax.plot(d[f"ts_1D"][var], d[f"ts_1D"][var].index, c="k", alpha=0.2, legend=False)
        ax.plot(d[f"ts_1MS"][var][:-1], d[f"ts_1MS"][var].index[:-1], c="k", ls="-", alpha=0.7, legend=False)
        d[f"ts_1YS"].iloc[-1] = d[f"ts_1MS"].iloc[-2].copy()
        ax.step(d[f"ts_1YS"][var], d[f"ts_1YS"][var].index, c="k", ls="--", lw=2, legend=False)

        if legend:
            ax.legend(
                [
                    Line2D([0], [0], color="k", alpha=a, ls=ls, label=l) for a, ls, l in zip(
                        [0.2, 0.7, 1], ["-", "-", "--"], ["Day", "Month", "Year"]
                    )
                ],
                ncols=1,
                loc="lr"
            )


def plot_volcano_activity(ax, df_unrest, df_eruptions, tlim=None, legend=True):
    if tlim:
        if len(df_unrest) != 0:
            df_unrest = df_unrest.loc[(df_unrest.DateEnd >= tlim[0]) & (df_unrest.DateStart <= tlim[1])]
        if len(df_eruptions) != 0:
            df_eruptions = df_eruptions.loc[(df_eruptions.DateEnd >= tlim[0]) & (df_eruptions.DateStart <= tlim[1])]

    for df, color in zip([df_unrest, df_eruptions], [("k", "gray5"), ("r", "red5")]):
        if len(df) == 0:
            continue

        # if tlim:
        #     df = df.loc[(df.DateEnd >= tlim[0]) & (df.DateStart <= tlim[1])]

        for ind in df.index:
            if df.loc[ind, "DateStart"] == df.loc[ind, "DateEnd"]:
                ax.plot(
                    [0, 1],
                    [df.loc[ind, "DateStart"], df.loc[ind, "DateStart"]],
                    c=color[0], linewidth=2
                )
            else:
                ax.fill_between(
                    [0, 1],
                    [df.loc[ind, "DateStart"], df.loc[ind, "DateStart"]],
                    [df.loc[ind, "DateEnd"], df.loc[ind, "DateEnd"]],
                    c=color[0]
                )
            for t in ["Start", "End"]:
                if df.loc[ind, f"Delta{t}"] != pd.to_timedelta(0, "D"):
                    ax.fill_between(
                        [0, 1],
                        [df.loc[ind, f"Date{t}"] - df.loc[ind, f"Delta{t}"],
                         df.loc[ind, f"Date{t}"] - df.loc[ind, f"Delta{t}"]],
                        [df.loc[ind, f"Date{t}"] + df.loc[ind, f"Delta{t}"],
                         df.loc[ind, f"Date{t}"] + df.loc[ind, f"Delta{t}"]],
                        c=color[1]
                    )

    if legend:
        if len(df_unrest) > 0 or len(df_unrest) > 0:
            ax.legend(
                [Line2D([0], [0], marker="s", ls="none", color=color, lw=2, label=label) for df, color, label in zip(
                    [df_unrest, df_eruptions], ["k", "r"], ["Unrest", "Eruption"]
                ) if len(df) > 0],
                loc="ll", ncols=1
            )
        else:
            ax.legend(
                [Line2D([0], [0], ls="none", color="k", marker="$*$", label="None")],
                loc="ll"
            )


def plot_ds(ax, ds, var, **kwargs):
    return ds[var].plot.imshow(
        ax=ax, x="distance", y="date", add_colorbar=False, add_labels=False, **kwargs
    )


def baselines2str(ds):
    dt = ds.date_dt.dt.days.values
    return (
        "[" + f"{round(np.nanmin(dt[dt != 9999]))}" +
        ", " + f"{round(np.nanmedian(dt[dt != 9999]))}" + f"({round(np.nanmean(dt[dt != 9999]))})" +
        ", " + f"{round(np.nanmax(dt[dt != 9999]))}]"
    )


def plot_single(ds, d_era5, df_unrest, df_eruptions, filepath="", **kwargs):
    vmax = ds.v.quantile(0.95).item()

    tlim = (str(np.datetime_as_string(ds.date.values.min(), unit="D")),
            str(np.datetime_as_string(ds.date.values.max(), unit="D")))

    fig, axs = uplt.subplots(nrows=1, ncols=7, figsize=(16, 8), spany=True, spanx=True, wratios=(3, 3, 3, 3, 1, 1, 1))
    vi = plot_ds(ax=axs[0], ds=ds, var="v", vmin=0, vmax=vmax, cmap="viridis")
    mi = plot_ds(ax=axs[1], ds=ds, var="dv_mean_vec", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[2], ds=ds, var="dv_median", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[3], ds=ds, var="dv_seasonal_vec", vmin=-1, vmax=1, cmap="RdBu_r")

    plot_era5(axT=axs[4], axP=axs[5], d=d_era5, tlim=tlim)
    plot_volcano_activity(ax=axs[6], df_unrest=df_unrest, df_eruptions=df_eruptions, tlim=tlim)

    fig.colorbar(
        vi, loc="b", col=1, length=0.8, extend="max", label="Velocity [m/yr]", labelsize="large", ticklabelsize="large"
    )
    fig.colorbar(
        mi, loc="b", col=3, length=0.8, extend="both", label="Normalised deviation",
        labelsize="large", ticklabelsize="large"
    )
    fig.colorbar(
        "coolwarm", loc="b", col=5, extend="neither", locator=[0, 0.5, 1],
        ticklabels=["Cold", "Mean", "Hot"], ticklabelsize="large"
    )
    fig.colorbar(
        "PuOr", loc="b", col=6, extend="neither", locator=[0, 0.5, 1],
        ticklabels=["Dry", "Mean", "Wet"], ticklabelsize="large"
    )
    if "date_dt" in ds.keys():
        axs[0].legend(
            Line2D([], [], label=baselines2str(ds=ds)),
            loc="lr", ncols=1, pad=0, handlelength=0, title="Baselines"
        )

    axs.format(
        abc=True, abcloc="ul", abcsize="large", abcbbox=True, abcbboxalpha=0.75,
        ylabel="Date", xlabel="Distance from terminus along flowline [km]",
        # leftlabels=("ITS_LIVE", "TICOI"),
        toplabels=(
            "Magnitude",
            # "Deviation from mean magnitude",
            "Deviation from mean vector",
            "Deviation from median",
            # "Deviation from seasonal magnitude",
            "Deviation from seasonal vector",
            "Temperature", "Precipitation", "Volcanic activity"
        ),
        ylocator="year", yminorlocator="month",
        labelsize="x-large", ticklabelsize="large",
    )
    axs[:4].format(**kwargs)

    axs[4].format(xlim=[-1, 1], xlocator=[-1, 0, 1])
    axs[5].format(xlim=[-1, 1], xlocator=[-1, 0, 1])
    axs[-1].format(xlim=[0, 1], xlocator="null")

    if filepath:
        fig.save(filepath)
        uplt.close(fig)
    else:
        uplt.show()

    # ds["v"].plot.imshow(ax=axs[0], x="distance", y="date", cmap="viridis", vmin=0, vmax=ds.v.quantile(0.95).values.item())
    # ds["dv_mean"].plot.imshow(ax=axs[1], x="distance", y="date", cmap="RdBu_r", vmin=-1, vmax=1)
    # ds["dv_mean_vec"].plot.imshow(ax=axs[2], x="distance", y="date", cmap="RdBu_r", vmin=-1, vmax=1)
    # ds["dv_median"].plot.imshow(ax=axs[3], x="distance", y="date", cmap="RdBu_r", vmin=-1, vmax=1)
    # ds["dv_seasonal"].plot.imshow(ax=axs[4], x="distance", y="date", cmap="RdBu_r", vmin=-1, vmax=1)
    # ds["dv_seasonal_vec"].plot.imshow(ax=axs[5], x="distance", y="date", cmap="RdBu_r", vmin=-1, vmax=1)

    # ds.resample(date="14D").mean()["v"].plot.imshow(ax=axs[3], x="distance", y="date", cmap="viridis", vmin=0, vmax=ds.v.quantile(0.95).values.item())

    # uplt.show()


def plot_itslive_ticoi(ds_i, ds_t, d_era5, df_unrest, df_eruptions, filepath="", **kwargs):
    vmax = max(ds_i.v.quantile(0.95).item(), ds_t.v.quantile(0.95).item())

    tlim = (str(np.datetime_as_string(ds_i.date.values.min(), unit="D")),
            str(np.datetime_as_string(ds_i.date.values.max(), unit="D")))

    fig, axs = uplt.subplots(nrows=2, ncols=7, figsize=(16, 8), spany=True, spanx=True, wratios=(3, 3, 3, 3, 1, 1, 1))

    vi = plot_ds(ax=axs[0, 0], ds=ds_i, var="v", vmin=0, vmax=vmax, cmap="viridis")
    # mi = plot_ds(ax=axs[0, 1], ds=ds_i, var="dv_mean", vmin=-1, vmax=1, cmap="RdBu_r")
    mi = plot_ds(ax=axs[0, 1], ds=ds_i, var="dv_mean_vec", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[0, 2], ds=ds_i, var="dv_median", vmin=-1, vmax=1, cmap="RdBu_r")
    # plot_ds(ax=axs[0, 4], ds=ds_i, var="dv_seasonal", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[0, 3], ds=ds_i, var="dv_seasonal_vec", vmin=-1, vmax=1, cmap="RdBu_r")

    plot_ds(ax=axs[1, 0], ds=ds_t, var="v", vmin=0, vmax=vmax, cmap="viridis")
    # plot_ds(ax=axs[1, 1], ds=ds_t, var="dv_mean", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[1, 1], ds=ds_t, var="dv_mean_vec", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[1, 2], ds=ds_t, var="dv_median", vmin=-1, vmax=1, cmap="RdBu_r")
    # plot_ds(ax=axs[1, 4], ds=ds_t, var="dv_seasonal", vmin=-1, vmax=1, cmap="RdBu_r")
    plot_ds(ax=axs[1, 3], ds=ds_t, var="dv_seasonal_vec", vmin=-1, vmax=1, cmap="RdBu_r")

    plot_era5(axT=axs[0, 4], axP=axs[0, 5], d=d_era5, tlim=tlim, legend=False)
    plot_era5(axT=axs[1, 4], axP=axs[1, 5], d=d_era5, tlim=tlim)
    plot_volcano_activity(ax=axs[0, 6], df_unrest=df_unrest, df_eruptions=df_eruptions, tlim=tlim, legend=False)
    plot_volcano_activity(ax=axs[1, 6], df_unrest=df_unrest, df_eruptions=df_eruptions, tlim=tlim)

    fig.colorbar(
        vi, loc="b", col=1, length=0.8, extend="max", label="Velocity [m/yr]", labelsize="large", ticklabelsize="large"
    )
    fig.colorbar(
        mi, loc="b", col=3, length=0.8, extend="both", label="Normalised deviation",
        labelsize="large", ticklabelsize="large"
    )
    fig.colorbar(
        "coolwarm", loc="b", col=5, extend="neither", locator=[0, 0.5, 1],
        ticklabels=["Cold", "Mean", "Hot"], ticklabelsize="large"
    )
    fig.colorbar(
        "PuOr", loc="b", col=6, extend="neither", locator=[0, 0.5, 1],
        ticklabels=["Dry", "Mean", "Wet"], ticklabelsize="large"
    )
    axs[0, 0].legend(
        Line2D([], [], label=baselines2str(ds=ds_i)),
        loc="lr", ncols=1, pad=0, handlelength=0, title="Baselines"
    )

    axs.format(
        abc=True, abcloc="ul", abcsize="large", abcbbox=True, abcbboxalpha=0.75,
        ylabel="Date", xlabel="Distance from terminus along flowline [km]",
        leftlabels=("ITS_LIVE", "TICOI"),
        toplabels=(
            "Magnitude",
            # "Deviation from mean magnitude",
            "Deviation from mean vector",
            "Deviation from median",
            # "Deviation from seasonal magnitude",
            "Deviation from seasonal vector",
            "Temperature", "Precipitation", "Volcanic activity"
        ),
        ylocator="year", yminorlocator="month",
        labelsize="x-large", ticklabelsize="large",
    )
    axs[:, :4].format(**kwargs)
    axs[0, :].format(abc="i.a")
    axs[1, :].format(abc="ii.a")

    axs[:, 4].format(xlim=[-1, 1], xlocator=[-1, 0, 1])
    axs[:, 5].format(xlim=[-1, 1], xlocator=[-1, 0, 1])
    axs[:, -1].format(xlim=[0, 1], xlocator="null")

    if filepath:
        fig.save(filepath)
        uplt.close(fig)
    else:
        uplt.show()
