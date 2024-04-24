from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import _compute_sw_global_threshold


def _align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])

    return axes


def plot_sw_gamma_single(
    ax: plt.Axes,
    epo_swa: np.ndarray,
    epo_gamma: np.ndarray,
    t_epoch_sws=2,
    ch_name=None,
    show=False,
) -> plt.Axes:
    """Plot SW-gamma averages around slow waves.

    Args:
        ax (plt.Axes): axis for plot.
        epo_swa (np.ndarray): SWA data, (n_sws, n_times).
        epo_gamma (np.ndarray): gamma data, (n_sws, n_times).
        t_epoch_sws (int, optional): Epoch time around SWs. Defaults to 2.
        ch_name (str, optional): Name of channel. Defaults to None, in which case no title is given.
        show (bool, optional): Defaults to False.

    Returns:
        plt.Axes: plotted axis.
    """

    # Get SWA data around SWs
    t_swa = np.linspace(-t_epoch_sws, t_epoch_sws, epo_swa.shape[1])
    epo_swa_mean = np.mean(epo_swa, axis=0)
    epo_swa_se = np.std(epo_swa, axis=0) / np.sqrt(len(epo_swa))

    # Get gamma data aroud SWs
    t_gamma = np.linspace(-t_epoch_sws, t_epoch_sws, epo_gamma.shape[1])
    epo_gamma_mean = np.mean(zscore(epo_gamma, axis=1), axis=0)
    epo_gamma_se = np.std(zscore(epo_gamma, axis=1), axis=0) / np.sqrt(len(epo_gamma))

    # Plot SWs
    line_sw = ax.plot(t_swa, epo_swa_mean, color="k", lw=1.5, label="Slow waves")
    ax.fill_between(
        t_swa,
        epo_swa_mean,
        epo_swa_mean - epo_swa_se,
        epo_swa_mean + epo_swa_se,
        alpha=0.3,
        color="k",
    )

    # Plot gamma
    ax2 = ax.twinx()
    line_gamma = ax2.plot(
        t_gamma, epo_gamma_mean, color="g", lw=1.5, label="Gamma power"
    )
    ax2.fill_between(
        t_gamma,
        epo_gamma_mean,
        epo_gamma_mean - epo_gamma_se,
        epo_gamma_mean + epo_gamma_se,
        alpha=0.3,
        color="g",
    )

    ax, ax2 = _align_yaxis(ax, ax2)

    # Plot axes lines
    ax.axvline(0, ls="--", color="k", lw=0.5)
    ax.axhline(0, ls="--", color="k", lw=0.5)

    # Title & labels
    if ch_name is not None:
        ax.set_title(ch_name, fontsize=15)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (uV)", fontsize=12)
    ax2.set_ylabel("Gamma power (a.u.)", fontsize=12)
    lns = line_sw + line_gamma
    ax.legend(lns, [l.get_label() for l in lns], fontsize=10)

    if show:
        plt.show()


def plot_sw_gamma(
    epo_swa: dict,
    epo_gamma: dict,
    # info_df: pd.DataFrame,
    # df_labels: pd.DataFrame,
    t_epoch_sws: float,
    figsize=(6, 4.5),
    show=False,
    save=True,
    save_path="",
    save_name="SW_gamma",
    save_format="pdf",
):
    """Plot SW-gamma averages for each channel, grand average and region average.

    Args:
        epo_swa (dict): Low-filtered data around SWs for each channel (keys).
        epo_gamma (dict): Gamma-filtered data around SWs for each channel (keys).
        info_df (pd.DataFrame): dataframe with channels metadata (good/bad).
        df_labels (pd.DataFrame): dataframe with channels labels.
        t_epoch_sws (float): s around SWs to plot.
        figsize (tuple, optional): Size of each subplot. Defaults to (6, 4.5).
        show (bool, optional): Whether to show the figure. Defaults to False.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_path (str, optional): Directory where to save the plot. Defaults to "".
        save_name (str, optional): Root name to save each plot. Defaults to "SW_gamma".
        save_format (str, optional): Format for saving figure. Defaults to "pdf".
    """
    # chs_all = info_df.index.to_list()
    # chs_good = info_df.index[info_df.bad == False].to_list()

    ###
    # All channels
    ###

    ch_names = list(epo_swa.keys())

    # Compute args for the grid plot
    n_cols = 8
    n_rows = len(ch_names) // n_cols
    if len(ch_names) % n_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        layout="constrained",
    )
    idx_ch = 0
    for col in range(n_cols):
        for row in range(n_rows):
            ch_name = ch_names[idx_ch]
            # ch_bad = ch_name not in chs_good

            # if not ch_bad:
            plot_sw_gamma_single(
                axs[row, col],
                epo_swa[ch_name],
                epo_gamma[ch_name],
                t_epoch_sws,
                show=False,
            )

            # Title
            # title = ch_name
            # title += " (BAD)" if ch_bad else ""
            axs[row, col].set_title(ch_name, fontsize=15)

            idx_ch += 1

            # # Split channels is blocks if provided
            # if blocks:
            #     if (row + 1) == blocks[col]:
            #         break

    if save:
        savepath = Path(save_path).joinpath(save_name + "_grid." + save_format)
        fig.savefig(savepath, bbox_inches="tight", format=save_format)
    if show:
        plt.show()

    ###
    # Grand average
    ###

    fig, ax = plt.subplots(1, 1, figsize=(figsize[0] * 2, figsize[1] * 2))
    # Compute averages
    # epo_swa = {k: v for k, v in epo_swa.items() if k in chs_good}
    avg_swa = np.array([np.mean(epo_ch, axis=0) for epo_ch in epo_swa.values()])
    # epo_gamma_good = {k: v for k, v in epo_gamma.items() if k in chs_good}
    avg_gamma = np.array(
        [np.mean(epo_ch, axis=0) for epo_ch in epo_gamma.values()]
    )
    plot_sw_gamma_single(
        ax,
        avg_swa,
        avg_gamma,
        t_epoch_sws,
        ch_name="Grand-average",
    )

    if save:
        savepath = Path(save_path).joinpath(save_name + "_grand_avg." + save_format)
        fig.savefig(savepath, bbox_inches="tight", format=save_format)
    if show:
        plt.show()

    plt.close()

def plot_sw_loc_glo_single(
    ax: plt.Axes,
    epo_swa: np.ndarray,
    sw_overlap: pd.DataFrame,
    thre_glo=None,
    t_epoch_sws=2,
    ch_name=None,
    show=False,
) -> plt.Axes:
    """Plot comparison of averaged local / global SWs.

    Args:
        ax (plt.Axes): axis for plot.
        epo_swa (np.ndarray): SWA data, (n_sws, n_times).
        sw_overlap (pd.DataFrame): Overlap of channel's SWs with those of other channels.
        thre_glo (float, optional): Proportion threshold to consider SW as global. Defaults to None.
        t_epoch_sws (int, optional): Epoch time around SWs. Defaults to 2.
        ch_name (str, optional): Name of channel. Defaults to None, in which case no title is given.
        show (bool, optional): Defaults to False.

    Returns:
        plt.Axes: plotted axis.
    """

    # Get SWA data around SWs
    t_swa = np.linspace(-t_epoch_sws, t_epoch_sws, epo_swa.shape[1])

    # Compute threshold and indexes from overlap
    if thre_glo is None:
        thre_glo = _compute_sw_global_threshold(sw_overlap)
    idx_loc = np.where(sw_overlap.mean(axis=1) < thre_glo)[0]
    idx_glo = np.where(sw_overlap.mean(axis=1) >= thre_glo)[0]

    # Get local / global SW epochs
    epo_loc_mean = np.mean(epo_swa[idx_loc], axis=0)
    epo_loc_se = np.std(epo_swa[idx_loc], axis=0) / np.sqrt(len(idx_loc))
    epo_glo_mean = np.mean(epo_swa[idx_glo], axis=0)
    epo_glo_se = np.std(epo_swa[idx_glo], axis=0) / np.sqrt(len(idx_glo))

    # Plot local SWs
    ax.plot(t_swa, epo_loc_mean, color="k", lw=1.5, label="Local")
    ax.fill_between(
        t_swa,
        epo_loc_mean,
        epo_loc_mean - epo_loc_se,
        epo_loc_mean + epo_loc_se,
        alpha=0.3,
        color="k",
    )
    # Plot global SWs
    ax.plot(t_swa, epo_glo_mean, color="b", lw=1.5, label="Global")
    ax.fill_between(
        t_swa,
        epo_glo_mean,
        epo_glo_mean - epo_glo_se,
        epo_glo_mean + epo_glo_se,
        alpha=0.3,
        color="b",
    )

    # Plot axes lines
    ax.axvline(0, ls="--", color="k", lw=0.5)
    ax.axhline(0, ls="--", color="k", lw=0.5)

    # Title & labels
    if ch_name is not None:
        ax.set_title(ch_name, fontsize=15)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (uV)", fontsize=12)
    ax.legend()

    if show:
        plt.show()

    return ax


def plot_sw_loc_glo(
    epo_swa: dict,
    sw_overlap: dict,
    # info_df: pd.DataFrame,
    t_epoch_sws: float,
    figsize=(6, 4.5),
    show=False,
    save=True,
    save_path="",
    save_name="SW_loc_glo",
    save_format="pdf",
):
    """Plot average loca / global SWs for each channel.

    Args:
        epo_swa (dict): Low-filtered data around SWs for each channel (keys).
        sw_overlap (dict): dict with SW overlap for each channel (keys).
        info_df (pd.DataFrame): dataframe with channels metadata (good/bad).
        t_epoch_sws (float): s around SWs to plot.
        figsize (tuple, optional): Size of each subplot. Defaults to (6, 4.5).
        show (bool, optional): Whether to show the figure. Defaults to False.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_path (str, optional): Directory where to save the plot. Defaults to "".
        save_name (str, optional): Root name to save each plot. Defaults to "SW_loc_glo".
        save_format (str, optional): Format for saving figure. Defaults to "pdf".
    """
    # chs_all = info_df.index.to_list()
    # chs_good = info_df.index[info_df.bad == False].to_list()

    ###
    # All channels
    ###

    ch_names = list(epo_swa.keys())

    # Compute args for the grid plot
    n_cols = 8
    n_rows = len(ch_names) // n_cols
    if len(ch_names) % n_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        layout="constrained",
    )
    idx_ch = 0
    for col in range(n_cols):
        for row in range(n_rows):
            ch_name = ch_names[idx_ch]
            # ch_bad = ch_name not in chs_good

            # if not ch_bad:
            axs[row, col] = plot_sw_loc_glo_single(
                axs[row, col],
                epo_swa[ch_name],
                sw_overlap[ch_name],
                t_epoch_sws=t_epoch_sws,
                show=False,
            )

            # Title
            # title = ch_name
            # title += " (BAD)" if ch_bad else ""
            axs[row, col].set_title(ch_name, fontsize=15)

            idx_ch += 1

            # # Split channels is blocks if provided
            # if blocks:
            #     if (row + 1) == blocks[col]:
            #         break

    if save:
        savepath = Path(save_path).joinpath(save_name + "_grid." + save_format)
        fig.savefig(savepath, bbox_inches="tight", format=save_format)
    if show:
        plt.show()

    plt.close()


def plot_sw_overlap(
    sw_overlap: dict,
    sw_delays: dict,
    show=False,
    save=True,
    save_path="",
    save_name="SW",
    save_format="pdf",
):
    """Plot overlap and delay matrices for SWs (ch x ch).

    Args:
        sw_overlap (dict): dict with SW overlap for each channel (keys).
        sw_delays (dict): dict with SW delays for each channel (keys).
        show (bool, optional): Whether to show the figure. Defaults to False.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_path (str, optional): Directory where to save the plot. Defaults to "".
        save_name (str, optional): Root name to save each plot. Defaults to "SW".
        save_format (str, optional): Format for saving figure. Defaults to "pdf".
    """
    # Get good channels
    chs_good = list(sw_overlap.keys())

    ###
    # Proportion of overlap plot
    ###

    sw_overlap_prop = pd.DataFrame(
        np.zeros((len(chs_good), len(chs_good))), index=chs_good, columns=chs_good
    )
    for ch in sw_overlap.keys():
        sw_overlap_prop.loc[ch, :] = sw_overlap[ch].mean(axis=0) * 100

    fig, ax = plt.subplots(figsize=(30, 20))
    sns.heatmap(sw_overlap_prop, annot=True, fmt=".1f", ax=ax)

    if save:
        savepath = Path(save_path).joinpath(save_name + "_overlap." + save_format)
        fig.savefig(savepath, bbox_inches="tight", format=save_format)

    if show:
        plt.show()

    ###
    # Time delays plot
    ###

    sw_overlap_del = pd.DataFrame(
        np.zeros((len(chs_good), len(chs_good))), index=chs_good, columns=chs_good
    )
    for ch in sw_delays.keys():
        sw_overlap_del.loc[ch, :] = sw_delays[ch].median(axis=0) * 1000

    fig, ax = plt.subplots(figsize=(30, 20))
    sns.heatmap(sw_overlap_del, annot=True, fmt=".0f", ax=ax, cmap="vlag")

    if save:
        savepath = Path(save_path).joinpath(save_name + "_delays." + save_format)
        fig.savefig(savepath, bbox_inches="tight", format=save_format)

    if show:
        plt.show()

    plt.close()
