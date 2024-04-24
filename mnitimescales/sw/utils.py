import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from scipy.stats import zscore
from yasa import sw_detect

###
# Functions for Slow Wave analysis
###


# def load_hypnogram(raw: mne.io.RawArray, mark_bads=True, bads_buffer=0.5) -> np.ndarray:
#     """Create hypnogram from raw annotations.

#     Args:
#         raw (mne.io.RawArray): MNE Raw object with sleep scoring and bads.
#         mark_bads (bool, optional): Whether to mark bad periods as artifacts. Defaults to True.
#         bads_buffer (int, optional): Periods around bads to mark as artifacts, in s. Defaults to 0.5.

#     Returns:
#         np.ndarray: hypnogram, with the same sfreq as raw.
#     """

#     # Mapping of stages
#     stage_code_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "bad": -1}

#     # Create container for stages
#     df_hypnogram = pd.DataFrame(columns=["t", "stage"])
#     df_hypnogram["t"] = raw.times
#     df_hypnogram["stage"] = -2

#     # Get relative onset of annotations
#     df_annots = raw.annotations.to_data_frame()
#     df_annots.drop_duplicates(inplace=True)
#     t0 = df_annots.loc[0, "onset"]
#     df_annots["onset_rel"] = [
#         (row["onset"] - t0).value / 1e9 for _, row in df_annots.iterrows()
#     ]

#     # Fill hypnogram with stages annotations
#     df_annots_stages = df_annots[
#         df_annots["description"].isin(["W", "N1", "N2", "N3", "R"])
#     ]
#     for _, row in df_annots_stages.iterrows():
#         # Get onset and duration in seconds
#         onset = row["onset_rel"]
#         duration = row["duration"]

#         # Fill hypnogram
#         df_hypnogram.loc[
#             (df_hypnogram["t"] >= onset) & (df_hypnogram["t"] < onset + duration),
#             "stage",
#         ] = stage_code_map[row["description"]]

#     # Mark bad periods
#     if mark_bads:
#         df_annots_bads = df_annots[
#             df_annots["description"].isin(["bad", "BAD_ACQ_SKIP"])
#         ]
#         for _, row in df_annots_bads.iterrows():
#             # Get onset and duration in seconds
#             onset = row["onset_rel"]
#             duration = row["duration"]

#             # Fill hypnogram
#             df_hypnogram.loc[
#                 (df_hypnogram["t"] >= onset - bads_buffer)
#                 & (df_hypnogram["t"] < onset + duration + bads_buffer),
#                 "stage",
#             ] = -1

#     return df_hypnogram["stage"].to_numpy()


def downsample_raw(raw):
    """We downsample the raw for speed, if not already downsampled."""

    raw_down = raw.copy()

    if raw_down.info["sfreq"] <= 128:
        return raw_down

    if raw.info["sfreq"] % 100 == 0:
        return raw_down.resample(sfreq=100)
    elif raw.info["sfreq"] % 128 == 0:
        return raw_down.resample(sfreq=128)

    # Unable to downsample
    return raw_down


def _epoch_sws(sw_events, data_raw, ch_names, sfreq, center, t_around=2):
    """Get data around slow waves."""

    # Seconds to timepoints
    t_around_tps = int(t_around * sfreq)

    # Select all indexes for each channel
    epo_sws_ch = {}
    for i, ch in enumerate(ch_names):
        sw_events_ch = sw_events[sw_events.Channel == ch].copy()
        idx_sws_ch = np.r_[
            [
                np.arange(int(t * sfreq - t_around_tps), int(t * sfreq + t_around_tps))
                for t in sw_events_ch[center]
            ]
        ]
        data_sws_ch = data_raw[i, idx_sws_ch].reshape(len(sw_events_ch), -1)
        epo_sws_ch[ch] = data_sws_ch

    return epo_sws_ch


def _check_swa_gamma(swa_mean, gamma_mean, sf_swa, sf_gamma):
    """Check if slow waves and gamma have the same polarity."""

    # Find minimum and zero-crossings
    idx_neg_peak = np.argmin(swa_mean)
    idx_zc_pn = np.where(np.diff(np.sign(swa_mean)) < 0)[0] + 1
    idx_zc_np = np.where(np.diff(np.sign(swa_mean)) > 0)[0] + 1

    # Indexes of negative and posite peaks
    idx_neg = np.arange(
        idx_zc_pn[idx_zc_pn < idx_neg_peak][-1], idx_zc_np[idx_zc_np > idx_neg_peak][0]
    )
    idx_pos_left = np.arange(idx_zc_np[idx_zc_np < idx_neg[0]][-1], idx_neg[0])
    idx_pos_right = np.arange(idx_neg[-1] + 1, idx_zc_pn[idx_zc_pn > idx_neg[-1]][0])

    # Check where gamma is higher
    f = int(sf_gamma / sf_swa)
    gamma_neg = np.mean(gamma_mean[idx_neg * f])
    gamma_pos_left = np.mean(gamma_mean[idx_pos_left * f])
    gamma_pos_right = np.mean(gamma_mean[idx_pos_right * f])

    if gamma_neg <= (gamma_pos_left + gamma_pos_right) / 2:
        return True
    else:
        return False


def _detect_sws(
    data: mne.io.RawArray,
    hypnogram: np.ndarray,
    stages=(2, 3),
    freq_sw=(0.5, 4),
    amp_ptp=(75, 350),
    dur_threshold=(0.8, 2),
    dur_neg=(0.1, 2),
    dur_pos=(0.1, 2),
    use_percentile=False,
    amp_percentile=25,
):
    """Detect Slow Waves in data, based on duration and amplitude criteria.

    Args:
        data (mne.io.RawArray): MNE Raw. Can be already filtered.
        hypnogram (np.ndarray): sleep stages of data, must have same sampling rate.
        stages (tuple, optional): Stages to include. Defaults to (2, 3).
        freq_sw (tuple, optional): Frequency filter (low, high), in Hz. Defaults to (0.5, 4).
        amp_ptp (tuple, optional): Amplitude thresholds (min, max), in uV. Defaults to (75, 350).
        dur_threshold (tuple, optional): Duration threshold (min, max), in s. Defaults to (0.5, 2).
        dur_neg (tuple, optional): Duration of negative peak (min, max), in s. Defaults to (0.1, 2).
        dur_pos (tuple, optional): Duration of positive peak (min, max), in s. Defaults to (0.1, 2).
        use_percentile (bool, optional): Whether to keep the higher % of found SWs. Defaults to False.
        amp_percentile (int, optional): higher % to use. Used if use_percentile is True. Defaults to 25.

    Returns:
        SWRes: class of yasa's results.
        sw_events: dataframe of detected SWs.
        filt_data: filtered data.
    """

    sfreq = data.info["sfreq"]
    ch_names = data.ch_names

    SWRes = sw_detect(
        data,
        sf=sfreq,
        ch_names=ch_names,
        freq_sw=freq_sw,
        dur_neg=dur_neg,
        dur_pos=dur_pos,
        amp_neg=(None, None),
        amp_pos=(None, None),
        amp_ptp=amp_ptp,
        hypno=hypnogram,
        include=stages,
        verbose=False,
    )

    filt_data = SWRes._data_filt
    sw_events = SWRes._events

    # Select SWs within duration threshold
    sw_events = sw_events.loc[
        (sw_events.Duration >= dur_threshold[0])
        & (sw_events.Duration <= dur_threshold[1])
    ]
    sw_events.reset_index(drop=True, inplace=True)
    SWRes._events = sw_events  # update events

    # Select SWs within percentile amplitude
    if use_percentile:
        sw_events_thres = []
        # Filter each channel separately
        for ch in ch_names:
            sw_events_ch = sw_events[sw_events.Channel == ch].copy()
            # Thresholds based on PTP amplitude
            ptp_thre = np.percentile(sw_events_ch.PTP, 100 - amp_percentile)
            ptp_max = np.percentile(
                sw_events_ch.PTP, 99
            )  # potential non-physiological events
            # Thresholds based on negative amplitude
            neg_thre = np.percentile(sw_events_ch.ValNegPeak, amp_percentile)
            neg_min = np.percentile(
                sw_events_ch.ValNegPeak, 1
            )  # potential non-physiological events
            sw_events_ch = sw_events_ch[
                (sw_events_ch.PTP >= ptp_thre)
                & (sw_events_ch.PTP < ptp_max)(sw_events_ch.ValNegPeak <= neg_thre)
                & (sw_events_ch.ValNegPeak > neg_min)
            ]
            sw_events_thres.append(sw_events_ch)
        # Concatenate back every channel
        sw_events = pd.concat(sw_events_thres, ignore_index=True)
        SWRes._events = sw_events  # update events

    return SWRes, sw_events, filt_data


def detect_sws_gamma(
    raw_swa: mne.io.RawArray,
    raw_gamma: mne.io.RawArray,
    hypnogram: np.ndarray,
    stages=(2, 3),
    dur_threshold=(0.8, 2),
    dur_neg=(0.1, 2),
    dur_pos=(0.1, 2),
    amp_ptp=(10, np.inf),
    use_percentile=True,
    amp_percentile=25,
    center_sws="NegPeak",
    t_epoch_sws=2,
):
    """Detect Slow Waves, making use of low frequency power (SWA) and gamma power.
    The techniques is used in: https://onlinelibrary.wiley.com/doi/10.1111/epi.13380
    The paramters for SWs are inspired from: https://www.nature.com/articles/s41593-023-01381-w

    Args:
        raw_swa (mne.io.RawArray): MNE Raw SWA filtered.
        raw_gamma (mne.io.RawArray): MNE Raw gamma filtered.
        hypnogram (np.ndarray): sleep stages of data, must have same sampling rate.
        stages (tuple, optional): Stages to include. Defaults to (2, 3).
        dur_threshold (tuple, optional): Duration threshold (min, max), in s. Defaults to (0.8, 2).
        amp_ptp (tuple, optional): Amplitude thresholds (min, max), in uV. Defaults to (10, np.inf).
        use_percentile (bool, optional): Whether to keep the higher % of found SWs. Defaults to True.
        amp_percentile (int, optional): higher % to use. Used if use_percentile is True. Defaults to 25.
        center_sws (str, optional): Center of slow waves epochs. Defaults to "NegPeak".
        t_epoch_sws (int, optional): Time around slow waves center, in s. Defaults to 2.

    Returns:
        sw_events (pd.DataFrame): dataframe with every detected slow wave.
        invert (dict): bool for channels' polarity inversion.
        epo_swa (dict): SWA data around each dtetcted slow wave, for each channel.
        epo_gamma(dict): gamma data around each dtetcted slow wave, for each channel.
    """

    # Raw info
    sfreq_swa = raw_swa.info["sfreq"]
    sfreq_gamma = raw_gamma.info["sfreq"]
    ch_names = raw_swa.ch_names

    # Compute slow waves for all channels with original polarity
    _, sw_events_orig, _ = _detect_sws(
        raw_swa,
        hypnogram,
        stages=stages,
        freq_sw=(None, None),
        amp_ptp=amp_ptp,
        dur_threshold=dur_threshold,
        dur_neg=dur_neg,
        dur_pos=dur_pos,
        use_percentile=use_percentile,
        amp_percentile=amp_percentile,
    )

    # Get SWA data around SWs
    data_swa = raw_swa.get_data()
    epo_swa = _epoch_sws(
        sw_events_orig,
        data_swa,
        ch_names,
        sfreq_swa,
        center=center_sws,
        t_around=t_epoch_sws,
    )
    epo_swa_mean = {ch: np.mean(e, axis=0) for ch, e in epo_swa.items()}

    # Get gamma data aroud SWs
    data_gamma = raw_gamma.get_data() ** 2  # square for power
    epo_gamma = _epoch_sws(
        sw_events_orig,
        data_gamma,
        ch_names,
        sfreq_gamma,
        center=center_sws,
        t_around=t_epoch_sws,
    )
    # z-score for better resolution
    epo_gamma_mean = {
        ch: np.mean(zscore(e, axis=1), axis=0) for ch, e in epo_gamma.items()
    }

    # Check consistency of SWA-gamma sign
    swa_gamma_polarity = {
        ch: _check_swa_gamma(
            epo_swa_mean[ch], epo_gamma_mean[ch], sfreq_swa, sfreq_gamma
        )
        for ch in ch_names
    }
    invert = {ch: not pol for ch, pol in swa_gamma_polarity.items()}

    # For channels with inconsistent polarity, flip sign
    raw_swa_flip = raw_swa.copy()
    chs_flip = [i for i, (ch, inv) in enumerate(invert.items()) if inv]
    raw_swa_flip._data[chs_flip] *= -1

    # Compute slow waves for all channels with original polarity
    _, sw_events, _ = _detect_sws(
        raw_swa_flip,
        hypnogram,
        stages=stages,
        freq_sw=(None, None),
        amp_ptp=amp_ptp,
        dur_threshold=dur_threshold,
        dur_neg=dur_neg,
        dur_pos=dur_pos,
        use_percentile=use_percentile,
        amp_percentile=amp_percentile,
    )

    # Re-compute SWA and gamma data around SWs
    data_swa = raw_swa_flip.get_data()
    epo_swa = _epoch_sws(
        sw_events,
        data_swa,
        ch_names,
        sfreq_swa,
        center=center_sws,
        t_around=t_epoch_sws,
    )
    data_gamma = raw_gamma.get_data() ** 2  # square for power
    epo_gamma = _epoch_sws(
        sw_events,
        data_gamma,
        ch_names,
        sfreq_gamma,
        center=center_sws,
        t_around=t_epoch_sws,
    )

    return sw_events, invert, epo_swa, epo_gamma


def sw_density(
    sw_events: pd.DataFrame, hypnogram: np.ndarray, ch_names: np.ndarray, sfreq: float
) -> pd.DataFrame:
    """Get slow waves density per stage and per channel.

    Args:
        sw_events (pd.DataFrame): dataframe with every detected slow wave.
        hypnogram (np.ndarray): Array with sleep stages.
        ch_names (np.ndarray): Channels names.
        sfreq (float): sampling frequency of hypnogram.

    Returns:
        pd.DataFrame: SWs density per stage and per channel.
    """

    # Get stages from sw_events
    stages = sw_events.Stage.unique()

    # Create container for sw density
    sw_density = pd.DataFrame(index=ch_names, columns=stages)

    for stage in stages:
        tps_in_stages = np.where(hypnogram == stage)[0].shape[0]
        minutes_in_stage = tps_in_stages / sfreq / 60

        for ch in ch_names:
            ch_sws = sw_events[(sw_events.Channel == ch) & (sw_events.Stage == stage)]
            sw_density.loc[ch, stage] = len(ch_sws) / minutes_in_stage

    return sw_density


def sw_conn(sw_events: pd.DataFrame, chs_good=None, sw_window=0.4):
    """Compute connectivity matrices through SWs measures

    Args:
        sw_events (pd.Dataframe): yasa's dataframe with SWs events
        chs_good (list, optional): list of good channels to consider. Defaults to None.
        sw_window (float, optional): window around SWs in s to consider for overlap and delays. Defaults to 0.4.

    Returns:
        dict: sw_overlap, each key is a channel, values are dataframes (n_SWs_ch, n_chs)
        with binary values of overlap of SWs between channels
        dict: sw_delays, each key is a channel, values are dataframes (n_SWs_ch, n_chs)
        with time delays of negative peaks between channels
    """
    chs_sws = sw_events.Channel.unique()
    if chs_good is not None:
        chs_sws = np.intersect1d(chs_sws, chs_good)

    # Dict for results
    sw_overlap = {}
    sw_delays = {}

    # Loop over channels as "seeds"
    for ch_seed in tqdm(chs_sws):
        # Get SWs for this channel
        sws_ch_seed = sw_events[sw_events.Channel == ch_seed]
        # Matrix for channel results
        sw_overlap_ch = np.zeros((len(sws_ch_seed), len(chs_sws)))
        sw_delays_ch = np.full_like(sw_overlap_ch, np.nan)
        # SWs times
        t_peak_seed = sws_ch_seed.NegPeak.to_numpy()
        # t_start = sws_ch_seed.Start.to_numpy().reshape(-1, 1)
        # t_end = sws_ch_seed.End.to_numpy().reshape(-1, 1)
        t_start = t_peak_seed - sw_window
        t_end = t_peak_seed + sw_window

        # Loop over all other channels
        for i, ch_targ in enumerate(chs_sws):
            if ch_targ == ch_seed:
                continue
            # Get SWs for this channel
            sws_ch_targ = sw_events[sw_events.Channel == ch_targ]
            # Check overlap of any SWs in ch_targ with SWs in ch_seed with broadcasting
            t_target = sws_ch_targ.NegPeak.to_numpy()
            mask_overlap = (t_target.reshape(1, -1) >= t_start.reshape(-1, 1)) & (
                t_target.reshape(1, -1) < t_end.reshape(-1, 1)
            )

            # Store results
            sw_overlap_seed_targ = mask_overlap.sum(axis=1).astype(int)
            # Make sure overlaps are "unique"
            sw_overlap_seed_targ[sw_overlap_seed_targ > 1] = 0
            sw_overlap_ch[:, i] = sw_overlap_seed_targ

            # Compute delay times
            # t_peak_targ = sws_ch_targ.NegPeak.to_numpy()
            idx_seed_delays = np.where(sw_overlap_seed_targ == 1)[0]
            idx_target_delays = np.where(mask_overlap[idx_seed_delays] == 1)[1]
            delays = t_peak_seed[idx_seed_delays] - t_target[idx_target_delays]
            sw_delays_ch[idx_seed_delays, i] = delays

        # Store results for seed channel
        sw_overlap_ch = pd.DataFrame(sw_overlap_ch, columns=chs_sws)
        sw_overlap_ch.drop(columns=ch_seed, inplace=True)
        sw_overlap[ch_seed] = sw_overlap_ch
        sw_delays_ch = pd.DataFrame(sw_delays_ch, columns=chs_sws)
        sw_delays_ch.drop(columns=ch_seed, inplace=True)
        sw_delays[ch_seed] = sw_delays_ch

    return sw_overlap, sw_delays


def _compute_sw_global_threshold(sw_overlap: dict) -> float:
    """Compute global threshold for SWs.

    Args:
        sw_overlap (dict): dict with overlap of SWs between channels.

    Returns:
        float: global threshold for SWs.
    """
    # Compute global threshold
    glo_thre = np.median(
        [np.median(swo.mean(axis=1).to_numpy()) for swo in sw_overlap.values()]
    )

    return glo_thre


# def format_sw_density_timecourse(
#     df_labels: pd.DataFrame,
#     df_bip: pd.DataFrame,
#     sw_events: pd.DataFrame,
#     sw_overlap: dict,
#     hypnogram: np.ndarray,
#     sfreq: float,
#     block_dur=30,
#     overlap_dur=15,
# ) -> pd.DataFrame:
#     """Format dataframe with SWs density .

#     Args:
#         df_labels (pd.DataFrame): labels and coordinates for each electrode.
#         df_bip (pd.DataFrame): bad electrodes for bipolar.
#         sw_events (pd.DataFrame): dataframe with every detected slow wave.
#         sw_overlap (dict): dict with overlap of SWs between channels.
#         hypnogram (np.ndarray): Array with sleep stages.
#         sfreq (float): sampling frequency of hypnogram.
#         block_dur (int, optional): length of block in seconds. Defaults to 30.
#         overlap_dur (int, optional): overlap between blocks in seconds. Defaults to 15.

#     Returns:
#         pd.DataFrame: dataframe with metric for each sleep stage.
#     """

#     # Define times of blocks (middle point)
#     times = np.arange(0, hypnogram.shape[0] / sfreq, block_dur - overlap_dur)

#     # Lists for building dataframe
#     times_df = []
#     chans = []
#     region = []
#     sw_density = []
#     sw_density_loc = []
#     sw_density_glo = []

#     # Get good channels
#     good_chans = [el for el in df_bip.index if el not in get_bad_channels(df_bip)]
#     lab_chans = [
#         convert_label(
#             lookup_bip_region(el, df_labels)[0],
#             lookup_bip_region(el, df_labels)[1],
#         )
#         for el in good_chans
#     ]
#     # Keep only good labels
#     lab_chans = {ch: lab for ch, lab in zip(good_chans, lab_chans) if lab != "NA"}
#     good_chans = list(lab_chans.keys())

#     # Calculate threshold for local / global SWs
#     glo_thre = _compute_sw_global_threshold(sw_overlap)

#     # Stages of SWs
#     Stages = sw_events.Stage.unique()

#     # Loop over time
#     for i, t in tqdm(enumerate(times), total=len(times)):
#         # Get SWs in block
#         sws_block = sw_events[
#             (sw_events.Channel.isin(good_chans))
#             & (sw_events.Start >= t)
#             & (sw_events.Start < t + block_dur)
#         ]

#         # Compute amout of "good" times
#         idx_good = np.where(
#             np.isin(hypnogram[int(t * sfreq) : int((t + block_dur) * sfreq)], Stages)
#         )[0]
#         s_good = len(idx_good) / sfreq  # in seconds

#         # Check is any SW was detected
#         if s_good == 0:
#             sw_density.extend([0] * len(good_chans))
#             sw_density_loc.extend([0] * len(good_chans))
#             sw_density_glo.extend([0] * len(good_chans))

#         elif len(sws_block) == 0:
#             sw_density.extend([0] * len(good_chans))
#             sw_density_loc.extend([0] * len(good_chans))
#             sw_density_glo.extend([0] * len(good_chans))

#         else:
#             # Loop over channels
#             for ch in good_chans:
#                 sws_ch = sw_events[sw_events.Channel == ch]
#                 # Get indexes of channel SWs in epoch
#                 idx_ch = np.where((sws_ch.Start >= t) & (sws_ch.Start < t + block_dur))[
#                     0
#                 ]
#                 # Total density per minute
#                 sw_density.append(len(idx_ch) / s_good * 60)
#                 # Get densities of local/global SWs
#                 n_loc = (sw_overlap[ch].iloc[idx_ch].mean(axis=1) < glo_thre).sum()
#                 n_glo = len(idx_ch) - n_loc
#                 sw_density_loc.append(n_loc / s_good * 60)
#                 sw_density_glo.append(n_glo / s_good * 60)

#         # Append info
#         times_df.extend([t + block_dur / 2] * len(good_chans))
#         chans.extend(good_chans)
#         region.extend([lab_chans[ch] for ch in good_chans])

#     df_density = pd.DataFrame(
#         np.c_[times_df, chans, region, sw_density, sw_density_loc, sw_density_glo],
#         columns=[
#             "time",
#             "chan",
#             "region",
#             "SW_density",
#             "SW_density_loc",
#             "SW_density_glo",
#         ],
#     )
#     # Convert to numeric
#     df_density[["time", "SW_density", "SW_density_loc", "SW_density_glo"]] = df_density[
#         ["time", "SW_density", "SW_density_loc", "SW_density_glo"]
#     ].apply(pd.to_numeric, errors="coerce")

#     return df_density
