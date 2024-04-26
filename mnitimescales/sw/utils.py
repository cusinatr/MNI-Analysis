import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from scipy.stats import zscore
from yasa import sw_detect


def load_hypnogram(raw: mne.io.RawArray, stage: str, bads_buffer=0.0) -> np.ndarray:
    """Create hypnogram with 'bad' annotations.

    Args:
        raw (mne.io.RawArray): MNE Raw object with bad periods.
        bads_buffer (int, optional): Periods around bads to mark as artifacts, in s. Defaults to 0.

    Returns:
        np.ndarray: hypnogram, with the same sfreq as raw.
    """

    # Mapping of stages
    stage_code_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "bad": -1}

    # Create container for stages
    df_hypnogram = pd.DataFrame(columns=["t", "stage"])
    df_hypnogram["t"] = raw.times
    df_hypnogram["stage"] = stage_code_map[stage]

    # Get relative onset of annotations
    df_annots = raw.annotations.to_data_frame()
    df_annots.drop_duplicates(inplace=True)
    df_annots["onset_rel"] = [
        row["onset"].value / 1e9 for _, row in df_annots.iterrows()
    ]

    # Mark bad periods
    df_annots_bads = df_annots[df_annots["description"].isin(["bad", "BAD_ACQ_SKIP"])]
    for _, row in df_annots_bads.iterrows():
        # Get onset and duration in seconds
        onset = row["onset_rel"]
        duration = row["duration"]

        # Fill hypnogram
        df_hypnogram.loc[
            (df_hypnogram["t"] >= onset - bads_buffer)
            & (df_hypnogram["t"] < onset + duration + bads_buffer),
            "stage",
        ] = -1

    return df_hypnogram["stage"].to_numpy()


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
            np.array(
                [
                    np.arange(
                        int(t * sfreq) - t_around_tps,
                        int(t * sfreq) + t_around_tps,
                        dtype=int,
                    )
                    for t in sw_events_ch[center]
                ]
            )
        ]
        data_sws_ch = data_raw[i, idx_sws_ch].reshape(len(sw_events_ch), -1)
        epo_sws_ch[ch] = data_sws_ch

    return epo_sws_ch


def _check_swa_gamma(swa_mean, gamma_mean, sf_swa, sf_gamma):
    """Check if slow waves and gamma have the same polarity."""

    # Find minimum and zero-crossings
    idx_mid = len(swa_mean) // 2
    idx_search = len(swa_mean) // 4
    idx_neg_peak = np.argmin(swa_mean[idx_mid - idx_search : idx_mid + idx_search]) + (
        idx_mid - idx_search
    )
    idx_zc_pn = np.where(np.diff(np.sign(swa_mean)) < 0)[0] + 1
    idx_zc_np = np.where(np.diff(np.sign(swa_mean)) > 0)[0] + 1

    # Indexes of negative and posite peaks
    idx_neg = np.arange(
        idx_zc_pn[idx_zc_pn < idx_neg_peak][-1], idx_zc_np[idx_zc_np > idx_neg_peak][0]
    )
    if idx_zc_np[idx_zc_np < idx_neg[0]].size != 0:
        idx_cross_left = idx_zc_np[idx_zc_np < idx_neg[0]][-1]
    else:
        idx_cross_left = 0
    idx_pos_left = np.arange(idx_cross_left, idx_neg[0])
    if idx_zc_pn[idx_zc_pn > idx_neg[-1]].size != 0:
        idx_cross_right = idx_zc_pn[idx_zc_pn > idx_neg[-1]][0]
    else:
        idx_cross_right = len(swa_mean)
    idx_pos_right = np.arange(idx_neg[-1] + 1, idx_cross_right)

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
        sw_events: dataframe of detected SWs.
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
            sw_events_ch = sw_events_ch[
                sw_events_ch.PTP >= ptp_thre
            ]
            sw_events_thres.append(sw_events_ch)
        # Concatenate back every channel
        sw_events = pd.concat(sw_events_thres, ignore_index=True)
        SWRes._events = sw_events  # update events

    return sw_events


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
    sw_events_orig = _detect_sws(
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
    data_gamma = raw_gamma.get_data()
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
    sw_events = _detect_sws(
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
    data_gamma = raw_gamma.get_data()
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
    """Get slow waves density per channel.

    Args:
        sw_events (pd.DataFrame): dataframe with every detected slow wave.
        hypnogram (np.ndarray): Array with sleep stages.
        ch_names (np.ndarray): Channels names.
        sfreq (float): sampling frequency of hypnogram.

    Returns:
        pd.DataFrame: SWs density per channel.
    """

    # Create dataframe for sw density
    sw_density = pd.DataFrame(index=ch_names, columns=["total", "local", "global"])

    # for stage in stages:
    tps_good = np.where(hypnogram != -1)[0].shape[0]
    minutes_good = tps_good / sfreq / 60.

    for ch in ch_names:
        ch_sws = sw_events[sw_events.Channel == ch]
        sw_density.loc[ch, "total"] = len(ch_sws) / minutes_good
        sw_density.loc[ch, "local"] = len(ch_sws[ch_sws.Global == 0]) / minutes_good
        sw_density.loc[ch, "global"] = len(ch_sws[ch_sws.Global == 1]) / minutes_good

    return sw_density


def sw_conn(sw_events: pd.DataFrame, chs_good=None, sw_window=0.3):
    """Compute connectivity matrices through SWs measures

    Args:
        sw_events (pd.Dataframe): yasa's dataframe with SWs events
        chs_good (list, optional): list of good channels to consider. Defaults to None.
        sw_window (float, optional): window around SWs in s to consider for overlap and delays. Defaults to 0.3.

    Returns:
        dict: sw_overlap, each key is a channel, values are dataframes (n_SWs_ch, n_chs)
        with binary values of overlap of SWs between channels
    """
    chs_sws = sw_events.Channel.unique()
    if chs_good is not None:
        chs_sws = np.intersect1d(chs_sws, chs_good)

    # Dict for results
    sw_overlap = {}

    # Loop over channels as "seeds"
    for ch_seed in tqdm(chs_sws):
        # Get SWs for this channel
        sws_ch_seed = sw_events[sw_events.Channel == ch_seed]
        # Matrix for channel results
        sw_overlap_ch = np.zeros((len(sws_ch_seed), len(chs_sws)))
        # SWs times
        t_peak_seed = sws_ch_seed.NegPeak.to_numpy()
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

        # Store results for seed channel
        sw_overlap_ch = pd.DataFrame(sw_overlap_ch, columns=chs_sws)
        sw_overlap_ch.drop(columns=ch_seed, inplace=True)
        sw_overlap[ch_seed] = sw_overlap_ch

    return sw_overlap


def compute_sw_global_threshold(sw_overlap: dict) -> float:
    """Compute global threshold for SWs.

    Args:
        sw_overlap (dict): dict with overlap of SWs between channels.

    Returns:
        float: global threshold for SWs.
    """
    # Compute global threshold
    glo_thre = np.mean(
        [np.median(swo.mean(axis=1).to_numpy()) for swo in sw_overlap.values()]
    )
    # Compute dict with binary values
    sw_glo_bool = {
        ch: sw_overlap[ch].mean(axis=1) >= glo_thre for ch in sw_overlap.keys()
    }

    return glo_thre, sw_glo_bool