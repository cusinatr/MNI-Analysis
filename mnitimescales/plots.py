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
import mnitimescales.utils as uti
from mnitimescales.spatial.utils import _exp_decay, _lin_curve

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

    TEXT_SIZE = 10
    MEANS_SIZE = 10
    TITLE_SIZE = 12
    TICK_SIZE = 10
    LABEL_SIZE = 12


def _set_font_params():
    """Set figures font"""

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = "Arial"
    rcParams["font.weight"] = "regular"


def _reset_default_rc():
    rcParams.update(rcParamsDefault)


def _get_figsize_inches(figsizes: tuple):

    # Conversion inches <-> cm
    cm = 1 / 2.54
    return [f * cm for f in figsizes]


def _format_spines(ax, s_inv=["top", "right"], s_bounds={}):
    """Format axis spines"""

    # Set spines to not visible
    for s in s_inv:
        ax.spines[s].set_visible(False)

    # Put bounds on spines
    for s, b in s_bounds.items():
        ax.spines[s].set_bounds(b[0], b[1])


def save_figure(fig, fig_path, format="svg"):
    """Save figure to file.

    Args:
        fig (plt.figure): figure to save.
        fig_path (str): path to save figure.
        format (str, optional): format to save figure. Defaults to "svg".
    """

    fig.savefig(
        fig_path,
        format=format,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )


def _get_fontsize_ratio(figsize: tuple, A_ref=72.25):
    """Compute the font size for a given figure size.

    Args:
        figsize (tuple): figure size in cm.
        A_ref (float, optional): reference area. Defaults to 72.25 (i.e. 8.5x8.5).

    Returns:
        float: ratio of font size to use.
    """

    A = figsize[0] * figsize[1]
    return np.sqrt(A / A_ref)


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

_set_font_params()


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
    reg_order=["OCX", "TCX", "PCX", "FCX", "ENT", "HIP", "AMY"],
    cond_order=["W", "N1", "N2", "N3", "R", ""],
    plot_type="bar",
    show_scatter=False,
    show_lines=True,
    markersize=4,
) -> plt.axes:
    """Bar or point plot of median metric for each region, diveded in conditions.

    Args:
        ax (plt.axes): axes to plot on.
        df_plot (pd.DataFrame): dataframe with region, condition and metric for each channel.
        jitter (float, optional): Jitter for scatter plots. Defaults to 0.05.
        sep (float, optional): Separation between conditions. Defaults to 0.3.
        colors_cond (dict, optional): dict with colors associated with each condition.
            Defaults to { "wake": "r", "rem": "g", "n3": "purple", "n2": "c", "n1": "dodgerblue"}.
        reg_order (list, optional): Order of regions to plot. Defaults to ["OCX", "TCX", "PCX", "FCX", "ENT", "HIP", "AMY"].
        cond_order (list, optional): Order of conditions to plot. Defaults to ["W", "N1", "N2", "N3", "R"].
        plot_type (str, optional): Type of plot. Can be "bar" or "point". Defaults to "bar".
        show_scatter (bool, optional): Show scatter plot. Defaults to False.
        show_lines (bool, optional): Show lines connecting points. Defaults to True.
        markersize (int, optional): Size of markers for point plot. Defaults to 4.

    Returns:
        plt.axes: modified axes.
    """

    # Define regions and conditions
    Regions = df_plot["region"].unique()
    Conds = [s for s in cond_order if s in df_plot["cond"].unique()]

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

        # Bar plot
        if plot_type == "bar":
            means_reg = []
            for j, cond in enumerate(Conds):
                mean_reg_cond = np.nanmedian(data_reg_conds[j])
                iqr = np.quantile(data_reg_conds[j], (0.25, 0.75))
                xerr = np.abs(iqr - mean_reg_cond).reshape(-1, 1)
                means_reg.append(mean_reg_cond)
                ax.barh(
                    ticks_regions[i] + ticks_conds[j],
                    mean_reg_cond,
                    height=sep,
                    xerr=xerr,
                    color=colors_cond[cond],
                    ecolor="k",
                    error_kw={"elinewidth": 1, "dash_capstyle": "round"},
                    linewidth=0,
                )
        elif plot_type == "point":
            # Median values & IQR
            means_reg = []
            for j, cond in enumerate(Conds):
                mean_reg_cond = np.nanmedian(data_reg_conds[j])
                iqr = np.quantile(data_reg_conds[j], (0.25, 0.75))
                means_reg.append(mean_reg_cond)
                ax.scatter(
                    mean_reg_cond,
                    ticks_regions[i] + ticks_conds[j],
                    c=colors_cond[cond],
                    marker="o",
                    s=markersize,
                    zorder=4,
                )
                ax.plot(
                    [iqr[0], iqr[1]],
                    [ticks_regions[i] + ticks_conds[j]] * 2,
                    c=colors_cond[cond],
                    lw=0.6,
                    solid_capstyle="round",
                    zorder=4,
                )
            if show_lines:
                for k in range(len(means_reg) - 1):
                    ax.plot(
                        [means_reg[k], means_reg[k + 1]],
                        [
                            ticks_regions[i] + ticks_conds[k],
                            ticks_regions[i] + ticks_conds[k + 1],
                        ],
                        lw=0.3,
                        color="k",
                        alpha=0.8,
                        zorder=3,
                    )

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

        i += 1  # keep track of region

    # Axes
    ax.set_yticks(
        ticks_regions,
        labels=[reg for reg in reg_order if reg in Regions],
        rotation=0,
        va="center",
        ha="right",
        fontsize=fsize.TICK_SIZE + 1,
    )
    ax.set_xlabel(metric_name, fontsize=fsize.LABEL_SIZE)

    # Spines
    _format_spines(ax)

    # Legend
    leg, leg_names = [], []
    for cond, color in colors_cond.items():
        if cond in Conds:
            leg.append(mpatches.Patch(color=color, alpha=1.0, linewidth=0))
            leg_names.append(cond)
    ax.legend(
        leg, leg_names, frameon=False, fontsize=fsize.TEXT_SIZE, loc="lower right"
    )


def plot_parcellated_metric(
    parc_metric: np.ndarray,
    parc_labels: np.ndarray,
    subjects_dir: str,
    labels_mne: list,
    log_scale=False,
    minmax=(None, None),
    zero_center=False,
    title="",
    cmap="inferno",
    label="Timescales [ms]",
    cbar_format="1f",
    cbar_ticks=None,
    figsize=(20, 15),
):
    """Plot parcellated metric on inflated brain.

    Args:
        parc_metric (np.ndarray): metric on brain parcels.
        parc_labels (np.ndarray): labels for the parcellation to use.
        subjects_dir (str): dir where surface data is (e.g. mne one).
        labels_mne (list): mne.Label object with info on region.
        log_scale (bool, optional): whether to use log scale for colorbar. Defaults to False.
        minmax (tuple, optional): min, max values for colorbar. Defaults to (None, None).
        zero_center (bool, optional): whether to center colorbar at 0. Defaults to False.
        title (str, optional): title for the plot. Defaults to "".
        cmap (str, optional): colormap. Defaults to "inferno".
        label (str, optional): label of the colorbar. Defaults to "Timescales [ms]".
        cbar_format (str): label format for colorbar. Defaults to "1f".
        cbar_ticks (list, optional): provided ticks for colorbar. Defaults to None.
        figsize (tuple): figure size. Defaults to (20, 15).

    Returns:
        figure, axes
    """

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
            thresh=0,
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
    # Add medial wall in grey
    lab_plot = [
        label
        for label in labels_mne
        if (("Unknown" in label.name[:-3]) or ("???" in label.name[:-3]))
    ][0]
    brain.add_label(lab_plot, borders=False, color="grey")

    brainviews = []
    brain.show_view("lat")
    brainviews.append(brain.screenshot())
    brain.show_view("med")
    brainviews.append(brain.screenshot())
    brain.close()

    fig, ax = plt.subplots(
        figsize=_get_figsize_inches(figsize), dpi=300
    )  # , layout="constrained")
    # Get fontsize for the figure
    img = ax.imshow(np.concatenate(brainviews, axis=1), cmap=cmap, norm=norm)
    cax = inset_axes(ax, width="50%", height="2%", loc=8, borderpad=3)
    cbar = fig.colorbar(
        img, cax=cax, orientation="horizontal", format="%." + cbar_format
    )
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    else:
        cbar.set_ticks([minv, maxv])
    cbar.set_label(label=label, size=32)
    cbar.mappable.set_clim(minv, maxv)
    cbar.ax.tick_params(labelsize=25)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=32)
    _format_spines(ax, s_inv=["top", "right", "bottom", "left"])

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
    """Bar plot with average timescales value."""

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


def half_violin_plot(
    ax: plt.Axes,
    y_data: float,
    x_pos: float,
    ci: list,
    y_boot: np.ndarray,
    color: str,
    pval=None,
    alpha=0.5,
    add_line=False,
):
    """Generate half violin plots with average and confidence intervals.

    Args:
        ax (plt.Axes): matlotlib axes to draw on.
        y_data (float): average value to plot as a point.
        x_pos (float): x position where to center the plot.
        ci (list): (lower, upper) conf. interval.
        y_boot (np.ndarray): bootstraps of the y value.
        color (str): color of the half violin. Defaults to str.
        pval (str, optional): pvalue to annotate. Defaults to None.
        alpha (float, optional): Alpha level of the half violin. Defaults to 0.5.
        add_line (bool, optional): whthwer to add a line at 0. Defaults to False.

    Returns:
        plt.Axes: modified axes.
    """

    # Plot point estimate
    ax.scatter(x=x_pos, y=y_data, color="k", s=36, zorder=10)
    # Plot confidence interval bar
    ax.plot([x_pos, x_pos], ci, color="k", lw=1.5)
    # Plot violin
    v = ax.violinplot(
        y_boot, positions=[x_pos], showmeans=False, showmedians=False, showextrema=False
    )
    # Restrict to right half
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
    # Add horizontal line at 0
    if add_line:
        ax.axhline(0, color="k", ls="--", lw=1, dashes=(6, 8))

    # Add pvalue if not None
    if pval is not None:
        ax.annotate(
            pval,
            xy=(x_pos, min(y_boot) * 0.95),
            xycoords="data",
            fontsize=fsize.TEXT_SIZE,
            ha="center",
        )

    return ax


def slope_plot(ax: plt.Axes, df_plot: pd.DataFrame):
    """Slope plot with values per region for different stages.

    Args:
        ax (plt.Axes): axis to plot on.
        df_plot (pd.DataFrame): dataframe with values per region per stage.

    Returns:
        plt.Axes: modified axis.
    """

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
    ax.set_xticks(xticks, labels=labs)
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

    return fig, ax


def plot_corr(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    p_corr: float,
    xy_annot=(0.7, 0.85),
    markersize=12,
    alpha=0.05,
    color="k",
    color_line=None,
    figsize=(6, 6),
    title="",
    xlabel="",
    ylabel="",
    xlims=None,
    ylims=None,
):
    """Plot correlation as scatter plot and correlation line
    with prediction interval.

    Args:
        ax (plt.Axes): axis to draw on.
        x (np.ndarray): x values.
        y (np.ndarray): y values.
        rho (float): correlation coefficient.
        p_corr (float): pvalue of correlation.
        xy_annot (tuple, optional): axis coordinates for annotating correlation value. Defaults to (0.7, 0.85).
        markersize (int, optional): size of scatter. Defaults to 12.
        alpha (float, optional): confidence level for prediction interval. Defaults to 0.05.
        color (str, optional): color of scatter. Defaults to "k".
        color_line (str, optional): color of correlation line. Defaults to None.
        figsize (tuple, optional): size of figure. Defaults to (6, 6).
        title (str, optional): plot title. Defaults to "".
        xlabel (str, optional): x axis label. Defaults to "".
        ylabel (str, optional): y axis label. Defaults to "".
        xlims (tuple, optional): x axis limits. Defaults to None.
        ylims (tuple, optional): y axis limits. Defaults to None.

    Returns:
        plt.Axes: modified axis
    """

    # Compute linear regression
    m, q, _, _, _ = linregress(x, y)

    # Plot
    if color_line is None:
        color_line = color
    x_plot = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, c=color, s=markersize, alpha=0.6, edgecolors="none")
    ax.plot(x_plot, m * x_plot + q, c=color_line, lw=2, ls="--")
    # Add 95% prediction interval to regression line
    t_fact = t.ppf(1 - alpha / 2, len(x) - 2)
    MSE = np.sum((y - (m * x + q)) ** 2) / (len(x) - 2)
    pi = t_fact * np.sqrt(
        MSE * (1 + 1 / len(x) + (x_plot - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    )
    ax.fill_between(
        x_plot, m * x_plot + q - pi, m * x_plot + q + pi, color="grey", alpha=0.05
    )

    # Plot parameters
    fontsize_fig = _get_fontsize_ratio(figsize)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="both", labelsize=fsize.TICK_SIZE)
    ax.set_xlabel(xlabel, fontsize=fsize.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)
    if xlims is not None:
        _format_spines(ax, s_bounds={"bottom": xlims})  # , "left": ylims})
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
    axs[0].set_ylabel("Value", fontsize=fsize.LABEL_SIZE)
    axs[0].set_title("Timescales across regions", fontsize=fsize.LABEL_SIZE)

    # Figure with relative values (to Wake)
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

    return fig, axs


def plot_sc_fit(
    data_stages: dict,
    params_stages: dict,
    colors_stage: dict,
    data_name="corr",
    ylabel="Max cross-correlation",
    dict_stages=None,
    figsize=(8, 8),
    dpi=300,
):
    """Plot exponential decay of SC across distance.

    Args:
        data_stages (dict): cross-correlation dataframe for each stage.
        params_stages (dict): fit parameters for each stage.
        colors_stage (dict): colors for each stage.
        data_name (str, optional): column in dataframe wit CC values. Defaults to "corr".
        ylabel (str, optional): y axis label. Defaults to "Max cross-correlation".
        dict_stages (dict, optional): stage names to plot. Defaults to None.
        figsize (tuple, optional): size of figure. Defaults to (8, 8).
        dpi (int, optional): figure resolution. Defaults to 300.

    Returns:
        figure, axes
    """

    gs_kw = dict(width_ratios=[2, 1])
    fig, axd = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "center right"], ["left", "lower right"]],
        gridspec_kw=gs_kw,
        figsize=_get_figsize_inches(figsize),
        dpi=dpi,
    )
    axs = [axd["left"], axd["upper right"], axd["center right"], axd["lower right"]]

    if list(params_stages.values())[0].size == 3:
        _fit_func = _exp_decay
    else:
        _fit_func = _lin_curve

    # First, plot with comparison between fits
    for stage in data_stages.keys():
        if dict_stages is not None:
            stage_title = dict_stages[stage]
        else:
            stage_title = stage
        axs[0].plot(
            data_stages[stage]["dist"].sort_values(),
            _fit_func(data_stages[stage]["dist"].sort_values(), *params_stages[stage]),
            "-",
            c=colors_stage[stage],
            label=stage_title,
            lw=2,
        )

    # Plot params
    axs[0].legend(frameon=False, fontsize=fsize.TEXT_SIZE)
    axs[0].set_xlim(0, 120)
    axs[0].set_xticks([0, 60, 120], labels=[0, 60, 120], fontsize=fsize.TICK_SIZE)
    axs[0].tick_params(axis="both", which="both", labelsize=fsize.TICK_SIZE)
    axs[0].set_xlabel("Distance [mm]", fontsize=fsize.LABEL_SIZE)
    axs[0].set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)
    _format_spines(axs[0])

    # Then, one subplot with data from all stages
    for i, stage in enumerate(data_stages.keys()):
        axs[i + 1].scatter(
            data_stages[stage]["dist"],
            data_stages[stage][data_name].abs(),
            c=colors_stage[stage],
            s=0.5,
            alpha=0.1,
            rasterized=True,
        )
        axs[i + 1].plot(
            data_stages[stage]["dist"].sort_values(),
            _fit_func(data_stages[stage]["dist"].sort_values(), *params_stages[stage]),
            "-",
            c="k",
            lw=2,
            zorder=9,
        )
        # Adjust plots
        axs[i + 1].set_xlim(0, 120)
        if (i + 1) == len(data_stages.keys()):
            axs[i + 1].set_xticks(
                [0, 60, 120], labels=[0, "", 120], fontsize=fsize.TICK_SIZE * 0.8
            )
        else:
            axs[i + 1].set_xticks([0, 60, 120], labels=[""] * 3, fontsize=0)
        axs[i + 1].set_ylim(0, 1)
        axs[i + 1].set_yticks([0, 1], labels=[0, 1], fontsize=fsize.TICK_SIZE * 0.8)
        _format_spines(axs[i + 1])

    return fig, axs


def plot_sc_fit_2(
    data_stages: dict,
    params_stages: dict,
    colors_stage: dict,
    data_name="corr",
    ylabel="Max cross-correlation",
    dict_stages=None,
    figsize=(8, 8),
    ylim=(0, 1),
    dpi=300,
):
    """
    As plot_sc_fit but with bigger figures per stage.
    """

    fig, axs = plt.subplots(
        3, 1, figsize=_get_figsize_inches(figsize), dpi=dpi, layout="constrained"
    )

    if list(params_stages.values())[0].size == 3:
        _fit_func = _exp_decay
    else:
        _fit_func = _lin_curve

    # Then, one subplot with data from all stages
    for i, stage in enumerate(data_stages.keys()):
        axs[i].scatter(
            data_stages[stage]["dist"],
            data_stages[stage][data_name].abs(),
            c=colors_stage[stage],
            s=0.5,
            alpha=0.1,
            rasterized=True,
        )
        axs[i].plot(
            data_stages[stage]["dist"].sort_values(),
            _fit_func(data_stages[stage]["dist"].sort_values(), *params_stages[stage]),
            "-",
            c="k",
            lw=2,
            zorder=9,
            label="exponential fit",
        )
        # Adjust plots
        axs[i].set_xlim(0, 120)
        if (i + 1) == len(data_stages.keys()):
            axs[i].set_xticks(
                [0, 60, 120], labels=[0, "", 120], fontsize=fsize.TICK_SIZE
            )
        else:
            axs[i].set_xticks([0, 60, 120], labels=[None] * 3, fontsize=0)
        axs[i].set_ylim(ylim)
        axs[i].set_yticks(ylim, labels=ylim, fontsize=fsize.TICK_SIZE)
        axs[i].set_title(dict_stages[stage], fontsize=fsize.TITLE_SIZE)
        _format_spines(axs[i])

    axs[0].legend(frameon=False, fontsize=fsize.TEXT_SIZE)
    axs[-1].set_xlabel("Distance [mm]", fontsize=fsize.LABEL_SIZE)
    axs[1].set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)

    return fig, axs


def plot_tc_sc_corr(
    ax: plt.Axes,
    df_rhos_d: pd.DataFrame,
    color="k",
    color_stars=None,
    label="",
    title="",
    xlabel="",
    ylabel="",
    ylims=None,
):
    """Plot correlation values of timescales and SC across distance bins.

    Args:
        ax (plt.Axes): axis to plot on.
        df_rhos_d (pd.DataFrame): correlation values across distance bins.
        color (str, optional): color for lines. Defaults to "k".
        color_stars (str, optional): color for significance markers. Defaults to None.
        label (str, optional): line label. Defaults to "".
        title (str, optional): plot title. Defaults to "".
        xlabel (str, optional): x axis label. Defaults to "".
        ylabel (str, optional): y axis label. Defaults to "".
        ylims (tuple, optional): y axis limits. Defaults to None.

    Returns:
        plt.Axes: modifies axis
    """

    ax.plot(df_rhos_d.index, df_rhos_d["rho"], lw=2, c=color, label=label)
    ax.fill_between(
        df_rhos_d.index,
        df_rhos_d["rho"] - df_rhos_d["rho_se"],
        df_rhos_d["rho"] + df_rhos_d["rho_se"],
        alpha=0.2,
        color=color,
    )
    ax.axhline(0.0, color="k", ls="--", lw=0.5)

    # Put significant distances
    sig_dist = df_rhos_d.index[df_rhos_d["pval"] < 0.05]
    if color_stars is None:
        color_stars = color
    ax.scatter(
        sig_dist,
        df_rhos_d.loc[sig_dist, "rho"],
        marker="*",
        c=color_stars,
        s=16,
        zorder=10,
    )

    # Plot parameters
    xlims = df_rhos_d.index[[0, -1]]
    delta = df_rhos_d.index[1] - df_rhos_d.index[0]
    ax.set_xlim(xlims[0] - delta / 2, xlims[1] + delta / 2)
    ax.set_ylim(ylims)
    ax.tick_params(axis="both", which="both", labelsize=fsize.TICK_SIZE)
    ax.set_xlabel(xlabel, fontsize=fsize.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=fsize.LABEL_SIZE)
    ax.set_title(title, fontsize=fsize.TITLE_SIZE)
    _format_spines(ax)

    return ax
