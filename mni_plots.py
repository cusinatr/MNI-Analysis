import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import pandas as pd
import numpy as np
from mne.viz import get_brain_class
from scipy.stats import linregress, t
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import plotly.graph_objects as go
from matplotlib import rcParams, rcParamsDefault, colors
import matplotlib.patches as mpatches
import mni_utils as uti

rcParams.update(rcParamsDefault)


###
# Helpers
###


class color:
    """Store plots objects' colors"""

    W = "r"
    N3 = "purple"
    R = "g"
    corr = "k"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class fsize:
    """Store plots objects' fontsizes"""

    TEXT_SIZE = 12
    MEANS_SIZE = 12
    TITLE_SIZE = 15
    TICK_SIZE = 10
    LABEL_SIZE = 12


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


def _get_camera_view_from_elevation_and_azimuth(elev, azim, r=1.5):
    """Compute plotly camera parameters from elevation and azimut."""
    # The radius is useful only when using a "perspective" projection,
    # otherwise, if projection is "orthographic",
    # one should tweak the "aspectratio" to emulate zoom
    # The camera position and orientation is set by three 3d vectors,
    # whose coordinates are independent of the plotted data.
    return {
        # Where the camera should look at
        # (it should always be looking at the center of the scene)
        "center": {"x": 0, "y": 0, "z": 0},
        # Where the camera should be located
        "eye": {
            "x": (
                r
                * math.cos(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "y": (
                r
                * math.sin(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "z": r * math.sin(elev / 360 * 2 * math.pi),
        },
        # How the camera should be rotated.
        # It is determined by a 3d vector indicating which direction
        # should look up in the generated plot
        "up": {
            "x": math.sin(elev / 360 * 2 * math.pi)
            * math.cos(azim / 360 * 2 * math.pi + math.pi),
            "y": math.sin(elev / 360 * 2 * math.pi)
            * math.sin(azim / 360 * 2 * math.pi + math.pi),
            "z": math.cos(elev / 360 * 2 * math.pi),
        },
        # "projection": {"type": "perspective"},
        "projection": {"type": "orthographic"},
    }


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

    _set_font_params()

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
    cbar_format="1f",
    cbar_ticks=None,
):
    """Plot parcellated metric on inflated brain.

    Args:
        parc_metric (np.ndarray): _description_
        parc_labels (np.ndarray): _description_
        subjects_dir (str): _description_
        log_scale (bool, optional): _description_. Defaults to False.
        minmax (tuple, optional): _description_. Defaults to (None, None).
        zero_center (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to "".
        cmap (str, optional): _description_. Defaults to "inferno".
        label (str, optional): _description_. Defaults to "Timescales [ms]".

    Returns:
        _type_: _description_
    """

    _set_font_params()

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
    cbar = fig.colorbar(
        img, cax=cax, orientation="horizontal", format="%." + cbar_format
    )
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(label=label, size=fsize.LABEL_SIZE)
    cbar.mappable.set_clim(minv, maxv)
    cbar.ax.tick_params(labelsize=fsize.TICK_SIZE)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)
    _format_spines(ax, s_inv=["top", "right", "bottom", "left"])

    _reset_default_rc()

    return fig, ax


def plot_hip_amy(
    surface_hip_amy,
    val_hip: float,
    val_amy: float,
    surface_nodes_labels: np.ndarray,
    lims_plot: list,
    cmap="inferno",
):
    """Plot a surface of Hippocampus and Amygdala colored according to a metric.

    Args:
        surface_hip_amy (nilearn.surface.surface.mesh): mesh of surface.
        val_hip (float): value to plot on hippocampus.
        val_amy (float): value to plot on amygdala.
        surface_nodes_labels (np.ndarray): label (0/1) of each vertex.
        lims_plot (list): limits of the plot.
        cmap (str, optional): colormap to use. Defaults to "inferno".

    Returns:
        plotly Figure
    """

    cmap = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(vmin=lims_plot[0], vmax=lims_plot[1])
    color_amy = cmap(norm(val_amy))
    color_hip = cmap(norm(val_hip))
    colors_plot = np.array([color_hip] * len(surface_hip_amy[0]))
    colors_plot[surface_nodes_labels == 1] = color_amy

    # First, create surface plot
    surf_trace = go.Mesh3d(
        x=surface_hip_amy.coordinates[:, 0],
        y=surface_hip_amy.coordinates[:, 1],
        z=surface_hip_amy.coordinates[:, 2],
        i=surface_hip_amy.faces[:, 0],
        j=surface_hip_amy.faces[:, 1],
        k=surface_hip_amy.faces[:, 2],
        color="white",
        vertexcolor=colors_plot,
        hoverinfo="skip",
    )

    fig = go.Figure(
        data=[surf_trace],
    )

    fig.update_layout(
        {
            "paper_bgcolor": "rgba(0,0,0,0)",  # transparent, to make it dark set a=0.8
        },
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.0)",
                color="white",
                zeroline=False,
                showticklabels=False,
                title_text="",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.0)",
                color="white",
                zeroline=False,
                showticklabels=False,
                title_text="",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.0)",
                color="white",
                zeroline=False,
                showticklabels=False,
                title_text="",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=_get_camera_view_from_elevation_and_azimuth(30, 100, r=10),
    )

    return fig


def bar_plot(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    y_lim=(None, None),
    y_label="Timescale [ms]",
    title="",
):
    """_summary_

    Args:
        ax (plt.Axes): _description_
        df_plot (pd.DataFrame): _description_
        y_lim (tuple, optional): _description_. Defaults to (None, None).
        y_label (str, optional): _description_. Defaults to "Timescale [ms]".
        title (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """

    _set_font_params()

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

    _reset_default_rc()

    return ax


def half_violin_plot(
    ax: plt.Axes,
    y_data: float,
    x_pos: float,
    ci: list,
    y_boot: np.ndarray,
    color: str,
    pval=None,
    alpha=0.5,
):
    """Generate half violin plots with average and confidence intervals.

    Args:
        ax (plt.Axes): matlotlib axes to draw on.
        y_data (float): average value to plot as a point.
        x_pos (float): x position where to center the plot.
        ci (list): (lower, upper) conf. interval.
        y_boot (np.ndarray): bootstraps of the y value.
        color (str): color of the half violin. Defaults to str.
        alpha (float, optional): Alpha level of the half violin. Defaults to 0.5.

    Returns:
        plt.Axes: modified axes.
    """

    _set_font_params()

    ax.scatter(x=x_pos, y=y_data, color="k", s=100, zorder=10)
    ax.plot([x_pos, x_pos], ci, color="k", lw=3)
    v = ax.violinplot(
        y_boot, positions=[x_pos], showmeans=False, showmedians=False, showextrema=False
    )
    for b in v["bodies"]:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(
            b.get_paths()[0].vertices[:, 0], m, np.inf
        )
        b.set_color(color)
        b.set_linewidth(0)
        b.set_alpha(alpha)

    # Add pvalue if not None
    if pval is not None:
        ax.annotate(
            pval,
            xy=(x_pos, min(y_boot) * 0.95),
            xycoords="data",
            fontsize=fsize.TICK_SIZE,
            ha="center"
        )

    _reset_default_rc()

    return ax


def slope_plot(ax: plt.Axes, df_plot: pd.DataFrame, ylabel=""):

    _set_font_params()

    # Extract labels
    labs = df_plot.columns.to_list()
    xticks = np.arange(len(labs))

    # Plot lines
    for _, row in df_plot.iterrows():
        for j in range(len(labs) - 1):
            ax.plot(
                [xticks[j], xticks[j + 1]],
                [row[labs[j]], row[labs[j + 1]]],
                c="grey",
                lw=0.5,
                alpha=0.5,
            )

    # Add average values
    avgs = [df_plot[col].mean() for col in labs]
    ses = [df_plot[col].sem() for col in labs]
    ax.errorbar(xticks, avgs, yerr=ses, fmt="ok", lw=2, capsize=4, capthick=2)
    # Add connecting line
    for j in range(len(labs) - 1):
        ax.plot(
            [xticks[j], xticks[j + 1]],
            [avgs[j], avgs[j + 1]],
            c="k",
            lw=1.5,
            alpha=0.8,
        )

    # Plot parameters
    ax.set_xticks(xticks, labels=labs, fontsize=fsize.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)
    _format_spines(ax)

    _reset_default_rc()

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
    """Plot MNI regions across stages.

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

    _set_font_params()

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
            fontsize=fsize.TEXT_SIZE,
        )
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)

    _reset_default_rc()

    return fig, ax


def plot_corr(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    p_corr: float,
    xy_annot=(0.7, 0.85),
    alpha=0.05,
    color="k",
    color_line=None,
    title="",
    xlabel="",
    ylabel="",
    xlims=None,
    ylims=None,
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

    _set_font_params()

    # Compute linear regression
    m, q, _, _, _ = linregress(x, y)

    # Plot
    if color_line is None:
        color_line = color
    x_plot = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, c=color, s=16, alpha=0.6)
    ax.plot(x_plot, m * x_plot + q, c=color_line, lw=2, ls="--")
    # Add 95% prediction interval to regression line
    t_fact = t.ppf(1 - alpha / 2, len(x) - 2)
    MSE = np.sum((y - (m * x + q)) ** 2) / (len(x) - 2)
    pi = t_fact * np.sqrt(
        MSE * (1 + 1 / len(x) + (x_plot - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    )
    ax.fill_between(
        x_plot, m * x_plot + q - pi, m * x_plot + q + pi, color="grey", alpha=0.06
    )

    # Plot parameters
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="both", labelsize=fsize.TICK_SIZE)
    ax.set_xlabel(xlabel, fontsize=fsize.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)
    if xlims is not None:
        _format_spines(ax, s_bounds={"bottom": xlims, "left": ylims})
    else:
        _format_spines(ax)

    # Annotate regression parameters
    p_str = r"p$_{\rm corr}$ "
    p_str += "= " + str(round(p_corr, 3)) if p_corr > 0.001 else "< 0.001"
    if p_corr < 0.05:
        p_str += r" $\bf{*}$"
    ax.annotate(
        r"$\rho$" + " = " + str(round(rho, 2)) + "\n" + p_str,
        xy=xy_annot,
        xycoords="axes fraction",
        fontsize=fsize.TEXT_SIZE,
    )

    _reset_default_rc()

    return ax


def plot_stages_diff(df_plot: pd.DataFrame, param: str, avg="mean"):

    _set_font_params()

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
    axs[0].set_ylabel("Value", fontsize=fsize.LABEL_SIZE)
    axs[0].set_title("Timescales across regions", fontsize=fsize.LABEL_SIZE)

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
    axs[1].set_ylabel("Value", fontsize=fsize.LABEL_SIZE)
    axs[1].set_title(f"Relative timescales to Wake", fontsize=fsize.TITLE_SIZE)

    _reset_default_rc()

    return fig, axs


def plot_sc_fit(
    data_stages: dict, params_stages: dict, colors_stage: dict, data_name="corr_max"
):

    _set_font_params()

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # One subplot with data from all stages
    for i, stage in enumerate(["W", "N3", "R"]):
        axs[i].plot(
            data_stages[stage]["dist"],
            data_stages[stage][data_name],
            "o",
            c=colors_stage[stage],
            ms=2,
            alpha=0.1,
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
        axs[i].set_xlabel("Distance [mm]", fontsize=fsize.LABEL_SIZE)
        axs[i].set_ylabel("Cross-correlation [a.u.]", fontsize=fsize.LABEL_SIZE)
        axs[i].set_title(f"Spatial correlation - {stage}", fontsize=fsize.TITLE_SIZE)
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
    axs[-1].set_xlabel("Distance [mm]", fontsize=fsize.LABEL_SIZE)
    axs[-1].set_ylabel("Cross-correlation [a.u.]", fontsize=fsize.LABEL_SIZE)
    axs[-1].set_title("Fit results", fontsize=fsize.TITLE_SIZE)
    _format_spines(axs[-1])

    _reset_default_rc()

    return fig, axs
