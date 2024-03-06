import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mne.viz import get_brain_class
from scipy.stats import linregress, spearmanr, pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from matplotlib import rcParams, rcParamsDefault, colors
import matplotlib.patches as mpatches
import mni_utils as uti

rcParams.update(rcParamsDefault)


###
# Helpers
###
class fsize:
    """Store plots objects' fontsizes"""

    TEXT_SIZE = 8
    MEANS_SIZE = 8
    TITLE_SIZE = 15
    TICK_SIZE = 8
    LABEL_SIZE = 10


def _set_font_params():
    """Set figures font"""

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = "Arial"
    rcParams["font.weight"] = "regular"


def _reset_default_rc():
    rcParams.update(rcParamsDefault)


def _format_spines(ax, s_inv=["top", "right"], s_bounds={}):
    """Format axis spines"""

    # Set spines to not visible
    for s in s_inv:
        ax.spines[s].set_visible(False)

    # Put bounds on spines
    for s, b in s_bounds.items():
        ax.spines[s].set_bounds(b[0], b[1])


###
# Plots
###


def plot_conds_regs(
    ax: plt.axes,
    df_plot: pd.DataFrame,
    jitter=0.05,
    sep=0.3,
    colors_cond={
        "W": "r",
        "R": "g",
        "N3": "purple",
        "N2": "c",
        "N1": "dodgerblue",
        "": "k",
    },
    reg_order=[
        "WM",
        "OC",
        "PC",
        "INS",
        "MTC",
        "STG",
        "BT",
        "ENT",
        "HIP",
        "AMY",
        "OFC",
        "LFC",
    ],
    conds_order=["W", "N1", "N2", "N3", "R", ""],
    show_scatter=True,
    show_lines=True,
    show_lines_means=True,
    text=False,
) -> plt.axes:
    """Scatterplot of metric for each region, diveded in conditions.

    Args:
        ax (plt.axes): axes to plot on.
        df_plot (pd.DataFrame): dataframe with region, condition and metric.
        jitter (float, optional): Jitter for scatter plots. Defaults to 0.05.
        sep (float, optional): Separation between conditions. Defaults to 0.3.
        colors_cond (dict, optional): dict with colors associated with each condition.
            Defaults to { "wake": "r", "rem": "g", "n3": "purple", "n2": "c", "n1": "dodgerblue"}.
        reg_order (list, optional): Order of regions to plot. Defaults to ["OCX", "TCX", "PCX", "FCX", "ENT", "HIP", "AMY"].
        conds_order (list, optional): Order of conditions to plot. Defaults to ["W", "N1", "N2", "N3", "R"].
        text (bool, optional): Whether to annotate text. Defaults to False.

    Returns:
        plt.axes: modified axes.
    """

    # Define regions and conditions
    Regions = df_plot["region"].unique()
    Conds = [s for s in conds_order if s in df_plot["cond"].unique()]

    # Metric name
    metric_name = df_plot.columns[-1]

    # Calculate ticks based on number of regions and conds
    n_conds = len(Conds)
    if n_conds % 2 == 0:  # conds are even
        ticks_conds = np.arange(0, n_conds * 2 + 1, 2, dtype=np.float64)
        ticks_conds -= n_conds - 1
        ticks_conds /= 2
    else:
        ticks_conds = np.arange(0, n_conds, dtype=np.float64)
        ticks_conds -= n_conds // 2
    ticks_conds *= sep
    if n_conds > 1:
        ticks_regions = np.arange(0, (n_conds - 1) * len(Regions), n_conds - 1)
    else:
        ticks_regions = np.arange(0, len(Regions))

    # Set fonts
    _set_font_params()

    i = 0
    # Loop through regions
    for reg in reg_order:
        if reg not in Regions:
            continue

        # Divide into conds
        data_reg_conds = [
            df_plot[(df_plot.region == reg) & (df_plot.cond == cond)][
                metric_name
            ].to_numpy(dtype=np.float64)
            for cond in Conds
        ]

        # Scatter plot
        if show_scatter:
            for j, cond in enumerate(Conds):
                x = ticks_regions[i] + jitter * np.random.randn(len(data_reg_conds[j]))
                ax.scatter(
                    x + ticks_conds[j],
                    data_reg_conds[j],
                    c=colors_cond[cond],
                    s=15,
                    alpha=0.6,
                    zorder=2,
                )
            if show_lines:
                # Connecting lines
                for k in range(len(data_reg_conds) - 1):
                    for j, (y1, y2) in enumerate(
                        zip(data_reg_conds[k], data_reg_conds[k + 1])
                    ):
                        if j < len(x):
                            ax.plot(
                                [x[j] + ticks_conds[k], x[j] + ticks_conds[k + 1]],
                                [y1, y2],
                                lw=0.3,
                                color="grey",
                                alpha=0.4,
                                zorder=1,
                            )

        # Mean values
        means_reg = []
        for j, cond in enumerate(Conds):
            mean_reg_cond = np.nanmedian(data_reg_conds[j])
            means_reg.append(mean_reg_cond)
            ax.scatter(
                ticks_regions[i] + (ticks_conds[j] - sep / 2),
                mean_reg_cond,
                c=colors_cond[cond],
                marker="o",
                s=50,
                zorder=4,
            )
            # Text
            if text:
                ax.annotate(
                    f"{mean_reg_cond:.1f}+-{np.std(data_reg_conds[j]):.1f}",
                    (
                        ticks_regions[i] + (ticks_conds[j] - sep / 2),
                        1.03 * np.max(data_reg_conds[j]),
                    ),
                    weight="bold",
                    ha="center",
                    va="center",
                    fontsize=fsize.TEXT_SIZE,
                    clip_on=True,
                )
        if show_lines_means:
            for k in range(len(means_reg) - 1):
                ax.plot(
                    [
                        ticks_regions[i] + (ticks_conds[k] - sep / 2),
                        ticks_regions[i] + (ticks_conds[k + 1] - sep / 2),
                    ],
                    [means_reg[k], means_reg[k + 1]],
                    lw=1.0,
                    color="k",
                    alpha=0.8,
                    zorder=3,
                )

        i += 1  # keep track of region

    # Axes
    ax.set_xticks(ticks_regions)
    ax.set_xticklabels(
        [reg for reg in reg_order if reg in Regions], rotation=45, ha="right"
    )
    ax.set_ylabel(metric_name, fontsize=fsize.LABEL_SIZE)
    ax.tick_params(axis="x", which="major", labelsize=fsize.LABEL_SIZE)
    ax.tick_params(axis="y", which="major", labelsize=fsize.TICK_SIZE)

    # Spines
    _format_spines(ax)

    # Legend
    leg, leg_names = [], []
    for cond, color in colors_cond.items():
        if cond in Conds:
            leg.append(mpatches.Patch(color=color, alpha=1.0, linewidth=0))
            leg_names.append(cond)
    ax.legend(leg, leg_names, frameon=False, fontsize=fsize.TEXT_SIZE)

    _reset_default_rc()

    return ax


def plot_parcellated_metric(
    parc_metric: np.ndarray,
    parc_labels: np.ndarray,
    subjects_dir: str,
    log_scale=False,
    minmax=(None, None),
    zero_center=False,
    title="",
    cmap="inferno",
    label="Timescales [ms]",
):
    Brain = get_brain_class()
    brain = Brain(
        "fsaverage",
        "lh",
        "inflated",
        subjects_dir=subjects_dir,
        cortex=(0.5, 0.5, 0.5),
        background="white",
        size=800,
    )

    # Colors to parcellations
    if zero_center:
        parc_data = np.hstack((0, parc_metric))
    else:
        parc_data = np.hstack((-1e6, parc_metric))

    # Set colormap
    if zero_center:
        mid = 0
        minv = -max(abs(minmax[0]), abs(minmax[1]))
        maxv = max(abs(minmax[0]), abs(minmax[1]))
        norm = colors.TwoSlopeNorm(0, vmin=minv, vmax=maxv)
    else:
        mid = None
        minv = minmax[0]
        maxv = minmax[1]
        norm = colors.Normalize()

    if log_scale:
        brain.add_data(
            array=np.log10(parc_data[parc_labels]),
            fmin=np.log10(minmax[0]),
            fmax=np.log10(minmax[1]),
            fmid=np.log10(mid),
            colormap=cmap,
            colorbar=False,
        )
    else:
        brain.add_data(
            array=parc_data[parc_labels],
            fmin=minv,
            fmax=maxv,
            fmid=mid,
            colormap=cmap,
            colorbar=False,
        )

    brainviews = []
    brain.show_view("lat")
    brainviews.append(brain.screenshot())
    brain.show_view("med")
    brainviews.append(brain.screenshot())
    brain.close()

    fig, ax = plt.subplots(figsize=[8, 6], layout="constrained")
    img = ax.imshow(np.concatenate(brainviews, axis=1), cmap=cmap, norm=norm)
    cax = inset_axes(ax, width="50%", height="2%", loc=8, borderpad=3)
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal", format="%.1f")
    cbar.set_label(label=label, size=fsize.LABEL_SIZE)
    cbar.mappable.set_clim(minv, maxv)
    sns.despine(fig, ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    return fig, ax


def bar_plot(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    y_lim=(None, None),
    y_label="Timescale [ms]",
    title="",
):
    ax.bar(range(len(df_plot)), df_plot["mean"].sort_values(), color="k", alpha=0.5)
    ax.errorbar(
        range(len(df_plot)),
        df_plot["mean"].sort_values(),
        yerr=df_plot["sem"][np.argsort(df_plot["mean"])],
        fmt="ok",
    )

    ax.set_ylabel(y_label, fontsize=fsize.LABEL_SIZE)
    ax.set_ylim(y_lim)
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(
        df_plot.index[np.argsort(df_plot["mean"])],
        rotation=45,
        ha="right",
        fontsize=fsize.LABEL_SIZE,
    )
    ax.tick_params(axis="y", which="both", labelsize=fsize.TICK_SIZE)
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)
    _format_spines(ax)

    return ax


def mni_plot(
    df_plot: pd.DataFrame,
    df_regions_len: pd.DataFrame,
    reg_order: list,
    conds_order: list,
    show_lines_means=True,
    sep=0.3,
    title="",
    ax=None,
):
    """Plo MNI regions across stages.

    Args:
        df_plot (pd.DataFrame): dataframe with timescales values.
        df_regions_len (pd.DataFrame): lenght of each macro region (to draw delimiters).
        reg_order (list): order of macro regions to plot.
        conds_order (list): order of conditions (stages) to plot.
        show_lines_means (bool, optional): Show lines connecting means. Defaults to True.
        sep (float, optional): Separation between consitions. Defaults to 0.3.
        title (str, optional): Defaults to "".
        ax (plt.Axes, optional): Axes to draw on. Defaults to None.

    Returns:
        fig, ax: Figure and Axes objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5), layout="constrained")
    else:
        fig = ax.get_figure()
    plot_conds_regs(
        ax,
        df_plot,
        reg_order=reg_order,
        conds_order=conds_order,
        show_lines=False,
        show_lines_means=show_lines_means,
        sep=sep,
    )
    # Add delimiters between lobes
    df_regions_len = pd.concat([pd.Series([0]), df_regions_len])
    for i in range(len(df_regions_len) - 1):
        x1, x2 = df_regions_len.iloc[i], df_regions_len.iloc[i + 1] - 1
        ax.axvline(x1 - 0.15, color="k", linewidth=0.5)
        ax.axvline(x2 + 0.15, color="k", linewidth=0.5)
        ax.text(
            (x1 + x2) / 2,
            ax.get_ylim()[1] * 0.95,
            df_regions_len.index[i + 1],
            ha="center",
            va="center",
            fontsize=10,
        )
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)

    return fig, ax


def plot_struct_corr(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    rand_obj,
    corr_type="spearman",
    stage="",
    color="k",
    title="",
):
    """Plot correlation of structure and timescales.

    Args:
        ax (plt.Axes): Axes to draw on.
        x (np.ndarray): x values.
        y (np.ndarray): y values.
        rand_obj (_type_): Null permutations object to calculate corrected p-value.
        corr_type (str, optional): Type of correlation. Can be pearson or spearman. Defaults to "spearman".
        stage (str, optional): Sleep stage. Defaults to "".
        color (str, optional): Defaults to "k".
        title (str, optional): Defaults to "".

    Returns:
        ax: plotted Axes object
    """

    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "pearson":
        corr_func = pearsonr

    rho, p = corr_func(x, y)

    # Compute corrected p-value
    msr_nulls = rand_obj.randomize(y)
    rho_null = np.array([corr_func(x, n_)[0] for n_ in msr_nulls])
    pv_perm = (abs(rho) < abs(rho_null)).sum() / msr_nulls.shape[0]
    print(
        f"Correlation between T1T2 and tau in {stage}: rho={rho}, p-corr={pv_perm} (p={p})"
    )

    # Plot
    ax.scatter(x, y, c=color, s=16, alpha=0.8)
    x_plot = np.linspace(x.min(), x.max(), 100)
    m, q, _, _, _ = linregress(x, y)
    ax.plot(x_plot, m * x_plot + q, c=color, lw=2, ls="--")
    ax.set_xlabel("T1w/T2w (hierarchy)", fontsize=15)
    ax.set_ylabel("Timescale [ms]", fontsize=15)
    ax.set_title(title)
    _format_spines(ax)

    # Add correlation values
    x_text, y_text = ax.get_xlim()[1] * 0.7, ax.get_ylim()[1] * 0.95
    ax.text(
        x_text, y_text, f"rho={rho:.2f}\np={p:.3f}\np_corr={pv_perm:.3f}", fontsize=10
    )

    return ax


def plot_corr_tau_sc(
    ax: plt.Axes,
    df_tau: pd.DataFrame,
    df_sc_params: pd.DataFrame,
    color="k",
    color_line=None,
    title="",
) -> plt.Axes:
    """_summary_

    Args:
        ax (plt.Axes): _description_
        df_tau (pd.DataFrame): _description_
        df_sc_params (pd.DataFrame): _description_
        color (str, optional): _description_. Defaults to "k".
        title (str, optional): _description_. Defaults to "".

    Returns:
        plt.Axes: _description_
    """

    # Make sure the rows of the two dataframes coincide
    y = df_sc_params.loc[df_tau.index].to_numpy().squeeze()
    x = df_tau.to_numpy().squeeze()

    # Get the slope and intercept of the linear regression
    res = linregress(x, y)
    print(f"Linregress results: r = {res[2]}, p = {res[3]}")
    print(f"Spearman results: r = {spearmanr(x, y)[0]}, p = {spearmanr(x, y)[1]}")
    intercept, slope = res[1], res[0]

    # Plot the data and the linear regression
    if color_line is None:
        color_line = color
    ax.scatter(x, y, c=color, alpha=0.5, s=36)
    x_plot = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_plot,
        intercept + slope * x_plot,
        c=color_line,
        ls="--",
        lw=3,
    )

    ax.set_ylabel("Spatial parameter", fontsize=15)
    ax.set_xlabel("Timescale [a.u.]", fontsize=15)
    ax.set_title(title)
    _format_spines(ax)

    # Add correlation values
    x_text, y_text = ax.get_xlim()[1] * 0.7, ax.get_ylim()[1] * 0.95
    ax.text(x_text, y_text, f"rho={res[2]:.2f}\np={res[3]:.3f}", fontsize=10)

    return ax


def plot_stages_diff(df_plot: pd.DataFrame, param: str, avg="mean"):

    # Figure with absolute values
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), layout="constrained")
    regions_order = (
        df_plot[df_plot["stage"] == "W"].sort_values(param)["region"].tolist()
    )
    sns.barplot(
        x="region",
        y=param,
        data=df_plot,
        hue="stage",
        order=regions_order,
        hue_order=["W", "N3", "R"],
        # estimator=avg,
        # errorbar=("ci", 95),
        errorbar=None,
        width=0.7,
        dodge=True,
        palette=["r", "purple", "g"],
        saturation=1,
        ax=axs[0],
    )
    axs[0].set_xticks(
        axs[0].get_xticks(), axs[0].get_xticklabels(), rotation=45, ha="right"
    )
    axs[0].set_ylabel("Value", fontsize=10)
    axs[0].set_title("Timescales across regions", fontsize=15)

    # Figure with relative values (to Wake)
    # TODO: adjust how statisctic is computed
    df_plot_rel = pd.DataFrame(columns=["region", "stage", param], index=df_plot.index)
    for region in df_plot["region"].unique():
        df_plot_rel.loc[df_plot["region"] == region, "region"] = region
        df_plot_rel.loc[df_plot["region"] == region, "stage"] = df_plot.loc[
            df_plot["region"] == region, "stage"
        ]
        df_plot_rel.loc[df_plot["region"] == region, param] = (
            df_plot.loc[df_plot["region"] == region, param]
            / df_plot.loc[df_plot["region"] == region, param].iloc[0]
        ) - 1
    df_plot_rel.drop(index=df_plot_rel[df_plot_rel["stage"] == "W"].index, inplace=True)
    sns.barplot(
        x="region",
        y=param,
        data=df_plot_rel,
        hue="stage",
        order=regions_order,
        hue_order=["N3", "R"],
        errorbar=None,
        width=0.7,
        dodge=True,
        palette=["purple", "g"],
        saturation=1,
        ax=axs[1],
    )
    axs[1].set_xticks(
        axs[1].get_xticks(), axs[1].get_xticklabels(), rotation=45, ha="right"
    )
    axs[1].set_ylabel("Value", fontsize=10)
    axs[1].set_title(f"Relative timescales to Wake", fontsize=15)

    return fig, axs


def plot_sc_fit(data_stages: dict, params_stages: dict, colors_stage: dict):

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # One subplot with data from all stages
    for i, stage in enumerate(["W", "N3", "R"]):
        axs[i].plot(
            data_stages[stage]["dist"],
            data_stages[stage]["corr_max"],
            "o",
            c=colors_stage[stage],
            ms=9,
            alpha=0.2,
        )
        axs[i].plot(
            data_stages[stage]["dist"].sort_values(),
            uti._exp_decay(
                data_stages[stage]["dist"].sort_values(), *params_stages[stage]
            ),
            "-",
            c="k",
            lw=2,
            zorder=9,
        )
        axs[i].set_xlabel("Distance", fontsize=10)
        axs[i].set_ylabel("Correlation", fontsize=10)
        axs[i].set_title(f"Spatial correlation - {stage}", fontsize=15)
        _format_spines(axs[i])

    # Last subplot with comparison between fits
    for stage in ["W", "N3", "R"]:
        axs[-1].plot(
            data_stages[stage]["dist"].sort_values(),
            uti._exp_decay(
                data_stages[stage]["dist"].sort_values(), *params_stages[stage]
            ),
            "-",
            c=colors_stage[stage],
            lw=2,
        )
    axs[-1].set_xlabel("Distance", fontsize=10)
    axs[-1].set_ylabel("Correlation", fontsize=10)
    axs[-1].set_title("Max correlation - fit all stages", fontsize=15)
    _format_spines(axs[-1])

    return fig, axs
