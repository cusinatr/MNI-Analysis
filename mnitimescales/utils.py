from pathlib import Path
import warnings
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy import signal
from scipy import optimize, spatial
from scipy.stats import pearsonr, spearmanr, bootstrap, zscore
from yasa import sw_detect
from pybispectra import compute_fft, TDE
import mne
import nibabel as nib
import statistics
import statsmodels.formula.api as smf
from sklearn.utils.validation import check_random_state


###
# General functions
###


def get_avg_tau_mni(data: pd.DataFrame, metric_name="tau", method="LME") -> pd.Series:
    """Get average timescale per parcel of the MNI atlas.
    Uses a mixed model to account for different patients.

    Args:
        data (pd.DataFrame): tau values, patients and regions.

    Returns:
        pd.Series: Aggregated tau values.
    """

    # Remove NaN if present
    data = data.dropna()

    # Mixed model
    if method == "LME":
        md = smf.mixedlm(metric_name + " ~ 0 + region", data, groups=data["pat"])
        mdf = md.fit()
        # Extract fitted parameters
        data_mni = mdf.fe_params
        data_mni.index = data_mni.index.str.strip("region[").str.strip("]")
    elif method == "median":
        data_mni = data.groupby("region")[metric_name].median()
    else:
        data_mni = data.groupby("region")[metric_name].mean()

    return data_mni


def convert_knee_tau(el_data: pd.Series) -> float:
    """Get timescale from knee fit, in milliseconds."""
    # Get knee and exponent
    knee = el_data["knee"]
    exp = el_data["exp"]

    # Knee frequency
    knee_freq = knee ** (1 / exp)

    return 1000 / (2 * np.pi * knee_freq)  # 1000 to convert to ms


def project_hemis_surf(surf, hemis="left"):
    """Keep brain surfaces of one hemisphere."""

    surf_hemis = deepcopy(surf)

    if hemis == "right":
        idx = np.where(surf.coordinates[:, 0] >= 0)[0]
    elif hemis == "left":
        idx = np.where(surf.coordinates[:, 0] <= 0)[0]
    idx_faces = [i for i, f in enumerate(surf.faces) if set(f).issubset(idx)]
    # We need to map the coordinates into new indexes values
    faces_hemis = surf.faces[idx_faces]
    mapper = {e: i for i, e in enumerate(idx)}

    surf_hemis = surf_hemis._replace(
        coordinates=surf.coordinates[idx], faces=np.vectorize(mapper.get)(faces_hemis)
    )

    return surf_hemis


def get_hip_amy_vtx(HO_atlas, surface_hip_amy, k=101):
    """Get vertices of Hippocampus and Amygdala on the surface.

    Args:
        HO_atlas (): Harvard-oxford atlas as returned from nilearn.
        surface_hip_amy (nilearn.surface.surface.mesh): Hippocampus/Amygdala mesh.
        k (int, optional): Number of nearest neighbours. Defaults to 101.

    Returns:
        np.ndarray: index of structure for each vertex.
    """

    # Get volume map of labels
    HO_map = HO_atlas["maps"].get_fdata()

    # Get labels for hippocampus and amygdala
    labels_hip_amy = [
        i
        for i, lab in enumerate(HO_atlas["labels"])
        if ("Left Hip" in lab) or ("Left Amy" in lab)
    ]
    idx_hip_amy = np.array(np.where(np.isin(HO_map, labels_hip_amy))).T
    idx_hip_amy_flat = HO_map[np.where(np.isin(HO_map, labels_hip_amy))].astype(
        np.int64
    )

    # Compute vertexes belonging to hippocampus and amygdala
    surface_nodes_labels = []
    for node in surface_hip_amy[0]:
        node_index = apply_affine(HO_atlas["maps"].affine, node, forward=False)
        dists = np.linalg.norm(idx_hip_amy - node_index, axis=1)
        # Mode of closes k nodes
        idx_closest_k = np.argsort(dists)[:k]
        label_closes_k = idx_hip_amy_flat[idx_closest_k]
        if labels_hip_amy[1] in label_closes_k:
            surface_nodes_labels.append(1)
        else:
            surface_nodes_labels.append(0)
    surface_nodes_labels = np.array(surface_nodes_labels)

    return surface_nodes_labels





###
# Functions for Slow Wave analysis
###


def load_hypnogram(raw: mne.io.RawArray, mark_bads=True, bads_buffer=0.5) -> np.ndarray:
    """Create hypnogram from raw annotations.

    Args:
        raw (mne.io.RawArray): MNE Raw object with sleep scoring and bads.
        mark_bads (bool, optional): Whether to mark bad periods as artifacts. Defaults to True.
        bads_buffer (int, optional): Periods around bads to mark as artifacts, in s. Defaults to 0.5.

    Returns:
        np.ndarray: hypnogram, with the same sfreq as raw.
    """

    # Mapping of stages
    stage_code_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "bad": -1}

    # Create container for stages
    df_hypnogram = pd.DataFrame(columns=["t", "stage"])
    df_hypnogram["t"] = raw.times
    df_hypnogram["stage"] = -2

    # Get relative onset of annotations
    df_annots = raw.annotations.to_data_frame()
    df_annots.drop_duplicates(inplace=True)
    t0 = df_annots.loc[0, "onset"]
    df_annots["onset_rel"] = [
        (row["onset"] - t0).value / 1e9 for _, row in df_annots.iterrows()
    ]

    # Fill hypnogram with stages annotations
    df_annots_stages = df_annots[
        df_annots["description"].isin(["W", "N1", "N2", "N3", "R"])
    ]
    for _, row in df_annots_stages.iterrows():
        # Get onset and duration in seconds
        onset = row["onset_rel"]
        duration = row["duration"]

        # Fill hypnogram
        df_hypnogram.loc[
            (df_hypnogram["t"] >= onset) & (df_hypnogram["t"] < onset + duration),
            "stage",
        ] = stage_code_map[row["description"]]

    # Mark bad periods
    if mark_bads:
        df_annots_bads = df_annots[
            df_annots["description"].isin(["bad", "BAD_ACQ_SKIP"])
        ]
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


def format_sw_density_timecourse(
    df_labels: pd.DataFrame,
    df_bip: pd.DataFrame,
    sw_events: pd.DataFrame,
    sw_overlap: dict,
    hypnogram: np.ndarray,
    sfreq: float,
    block_dur=30,
    overlap_dur=15,
) -> pd.DataFrame:
    """Format dataframe with SWs density .

    Args:
        df_labels (pd.DataFrame): labels and coordinates for each electrode.
        df_bip (pd.DataFrame): bad electrodes for bipolar.
        sw_events (pd.DataFrame): dataframe with every detected slow wave.
        sw_overlap (dict): dict with overlap of SWs between channels.
        hypnogram (np.ndarray): Array with sleep stages.
        sfreq (float): sampling frequency of hypnogram.
        block_dur (int, optional): length of block in seconds. Defaults to 30.
        overlap_dur (int, optional): overlap between blocks in seconds. Defaults to 15.

    Returns:
        pd.DataFrame: dataframe with metric for each sleep stage.
    """

    # Define times of blocks (middle point)
    times = np.arange(0, hypnogram.shape[0] / sfreq, block_dur - overlap_dur)

    # Lists for building dataframe
    times_df = []
    chans = []
    region = []
    sw_density = []
    sw_density_loc = []
    sw_density_glo = []

    # Get good channels
    good_chans = [el for el in df_bip.index if el not in get_bad_channels(df_bip)]
    lab_chans = [
        convert_label(
            lookup_bip_region(el, df_labels)[0],
            lookup_bip_region(el, df_labels)[1],
        )
        for el in good_chans
    ]
    # Keep only good labels
    lab_chans = {ch: lab for ch, lab in zip(good_chans, lab_chans) if lab != "NA"}
    good_chans = list(lab_chans.keys())

    # Calculate threshold for local / global SWs
    glo_thre = _compute_sw_global_threshold(sw_overlap)

    # Stages of SWs
    Stages = sw_events.Stage.unique()

    # Loop over time
    for i, t in tqdm(enumerate(times), total=len(times)):
        # Get SWs in block
        sws_block = sw_events[
            (sw_events.Channel.isin(good_chans))
            & (sw_events.Start >= t)
            & (sw_events.Start < t + block_dur)
        ]

        # Compute amout of "good" times
        idx_good = np.where(
            np.isin(hypnogram[int(t * sfreq) : int((t + block_dur) * sfreq)], Stages)
        )[0]
        s_good = len(idx_good) / sfreq  # in seconds

        # Check is any SW was detected
        if s_good == 0:
            sw_density.extend([0] * len(good_chans))
            sw_density_loc.extend([0] * len(good_chans))
            sw_density_glo.extend([0] * len(good_chans))

        elif len(sws_block) == 0:
            sw_density.extend([0] * len(good_chans))
            sw_density_loc.extend([0] * len(good_chans))
            sw_density_glo.extend([0] * len(good_chans))

        else:
            # Loop over channels
            for ch in good_chans:
                sws_ch = sw_events[sw_events.Channel == ch]
                # Get indexes of channel SWs in epoch
                idx_ch = np.where((sws_ch.Start >= t) & (sws_ch.Start < t + block_dur))[
                    0
                ]
                # Total density per minute
                sw_density.append(len(idx_ch) / s_good * 60)
                # Get densities of local/global SWs
                n_loc = (sw_overlap[ch].iloc[idx_ch].mean(axis=1) < glo_thre).sum()
                n_glo = len(idx_ch) - n_loc
                sw_density_loc.append(n_loc / s_good * 60)
                sw_density_glo.append(n_glo / s_good * 60)

        # Append info
        times_df.extend([t + block_dur / 2] * len(good_chans))
        chans.extend(good_chans)
        region.extend([lab_chans[ch] for ch in good_chans])

    df_density = pd.DataFrame(
        np.c_[times_df, chans, region, sw_density, sw_density_loc, sw_density_glo],
        columns=[
            "time",
            "chan",
            "region",
            "SW_density",
            "SW_density_loc",
            "SW_density_glo",
        ],
    )
    # Convert to numeric
    df_density[["time", "SW_density", "SW_density_loc", "SW_density_glo"]] = df_density[
        ["time", "SW_density", "SW_density_loc", "SW_density_glo"]
    ].apply(pd.to_numeric, errors="coerce")

    return df_density





###
# Functions for Spatial Correlations
###

def _check_hemi(mni_x: float):
    if mni_x < 0:
        return "left"
    else:
        return "right"

# TODO: rewrite function by decomposing tasks
def compute_SC(
    data: np.ndarray,
    df_info: pd.DataFrame,
    df_regions: pd.DataFrame,
    sfreq: int,
    path_mmp: Path,
    freq_band=False,
    band_freqs=[40, 80],
    use_bispectrum=False,
) -> pd.DataFrame:
    """Compute spatial correlation (SC) for every pair of channels.

    Args:
        data (np.ndarray): array of data, with shape (n_channels, n_times)
        df_info (pd.DataFrame): information about patients and channels
        df_regions (pd.DataFrame): map from region names to lobes
        sfreq (int): sampling frequency of signal
        freq_band (bool, optional): whether to compute SC in a specific frequency band. Defaults to False.
        band_freqs (list, optional): frequency range for frequency band. Defaults to [40, 80].

    Returns:
        pd.DataFrame: SC values (0 and max lag correlation) for every pair of channels.
    """
    pat_list = []
    ch1_list = []
    lobe1_list = []
    reg1_list = []
    mmp1_list = []
    ch2_list = []
    lobe2_list = []
    reg2_list = []
    mmp2_list = []
    dist = []
    corr_0 = []
    corr_max = []
    lag_max = []

    for pat in df_info["pat"].unique():
        print(pat)  # , end="|")

        data_pat = data[df_info["pat"] == pat]

        # Convert to mne Raw
        df_info_pat = df_info[df_info["pat"] == pat].reset_index(drop=True)
        raw = create_RawMNE(data_pat, df_info_pat["chan"].tolist(), sfreq)
        if raw is None:
            continue

        epo = create_epo(raw, freq_band, band_freqs)

        # Check MMP regions
        mmp_regs = mni_to_mmp(
            df_info_pat.loc[:, ["mni_x", "mni_y", "mni_z"]].to_numpy(dtype=float),
            path_mmp,
        )

        # Loop on the "seed" channel
        for i, ch1 in enumerate(raw.ch_names):
            data_ch1 = epo.get_data(picks=ch1)

            # Get MNI coordinates
            mni1 = df_info_pat.loc[i, ["mni_x", "mni_y", "mni_z"]].to_numpy(dtype=float)

            # Loop on the "target" channel
            for j in range(i + 1, len(raw.ch_names)):
                ch2 = raw.ch_names[j]

                # Get MNI coordinates
                mni2 = df_info_pat.loc[j, ["mni_x", "mni_y", "mni_z"]].to_numpy(
                    dtype=float
                )

                if ch1 == ch2:
                    continue

                # Retain only pairs on the same hemisphere
                hemi1, hemi2 = _check_hemi(mni1[0]), _check_hemi(mni2[0])
                if hemi1 != hemi2:
                    continue

                # Register info on electrodes location
                pat_list.append(pat)
                ch1_list.append(ch1)
                reg1_list.append(df_info_pat.loc[i, "region"])
                lobe1_list.append(
                    df_regions[df_regions["Region name"] == reg1_list[-1]][
                        "Lobe"
                    ].values[0]
                )
                mmp1_list.append(mmp_regs[i])
                ch2_list.append(ch2)
                reg2_list.append(df_info_pat.loc[j, "region"])
                lobe2_list.append(
                    df_regions[df_regions["Region name"] == reg2_list[-1]][
                        "Lobe"
                    ].values[0]
                )
                mmp2_list.append(mmp_regs[j])
                dist.append(np.linalg.norm(mni1 - mni2))

                data_ch2 = epo.get_data(picks=ch2)

                # Use "classic" cross-correlation or bispectrum
                if not use_bispectrum:
                    data_ch1 = data_ch1.squeeze()
                    data_ch2 = data_ch2.squeeze()
                    # Loop over epochs
                    corr_avg = []
                    for k in range(len(data_ch1)):
                        ch1_norm = (data_ch1[k] - data_ch1[k].mean()) / np.linalg.norm(
                            data_ch1[k]
                        )
                        ch2_norm = (data_ch2[k] - data_ch2[k].mean()) / np.linalg.norm(
                            data_ch2[k]
                        )
                        corr = signal.correlate(ch1_norm, ch2_norm, mode="same")
                        lags = signal.correlation_lags(
                            len(ch1_norm), len(ch2_norm), mode="same"
                        )
                        corr_avg.append(corr)
                    corr_avg = np.array(corr_avg).mean(axis=0)
                    corr_0.append(corr_avg[lags == 0][0])
                    corr_max.append(np.max(np.abs(corr_avg)))
                    lag_max.append(lags[np.argmax(np.abs(corr_avg))])
                else:
                    # Compute and use bispectrum
                    fft_coeffs, freqs = compute_fft(
                        data=np.concatenate([data_ch1, data_ch2], axis=1),
                        sampling_freq=float(sfreq),
                        n_points=2 * data_ch1.shape[-1]
                        + 1,  # recommended for time delay estimation
                        window="hamming",
                        verbose=False,
                    )
                    # initialise TDE object
                    tde = TDE(
                        data=fft_coeffs,
                        freqs=freqs,
                        sampling_freq=float(sfreq),
                        verbose=False,
                    )
                    if not freq_band:
                        band_freqs = (0, sfreq / 2)
                    # compute TDE
                    tde.compute(
                        indices=((0,), (1,)),
                        fmin=(band_freqs[0]),
                        fmax=(band_freqs[1]),
                        method=1,
                        antisym=True,
                    )
                    tde_times = tde.results.times
                    tde_results = tde.results.get_results()  # return results as array
                    # get max correlation and lag
                    corr_0.append(tde_results[0, 0, tde_times == 0][0])
                    corr_max.append(np.max(tde_results))
                    lag_max.append(tde_times[np.argmax(tde_results)])

    df_sc = pd.DataFrame(
        columns=[
            "ch_1",
            "ch_2",
            "lobe_1",
            "lobe_2",
            "region_1",
            "region_2",
            "mmp_1",
            "mmp_2",
            "dist",
            "corr_0",
            "corr_max",
            "lag_max",
        ]
    )
    df_sc["pat"] = pat_list
    df_sc["ch_1"] = ch1_list
    df_sc["ch_2"] = ch2_list
    df_sc["region_1"] = reg1_list
    df_sc["region_2"] = reg2_list
    df_sc["lobe_1"] = lobe1_list
    df_sc["lobe_2"] = lobe2_list
    df_sc["mmp_1"] = mmp1_list
    df_sc["mmp_2"] = mmp2_list
    df_sc["dist"] = dist
    df_sc["corr_0"] = corr_0
    df_sc["corr_max"] = corr_max
    df_sc["lag_max"] = lag_max

    return df_sc


def compute_sc_bin(df_sc: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    """Compute binned spatial correlations.

    Args:
        df_sc (pd.DataFrame): dataframe as computed with compute_SC function.
        bins (np.ndarray): bins for distance.

    Returns:
        pd.DataFrame: binned spatial correlations.
    """
    # Create bins from distance
    bins_cat, bins_values = pd.cut(df_sc["dist"], bins=bins, retbins=True)

    # Create new dataframes for binned spatial correlations
    df_sc_bin_avg = pd.DataFrame(columns=["dist", "bin", "corr_0", "corr_max"])
    df_sc_bin_sem = pd.DataFrame(columns=["dist", "corr_0_sem", "corr_max_sem"])
    df_sc_bin_avg["dist"] = bins_cat
    df_sc_bin_sem["dist"] = bins_cat

    # Compute average and sem for each bin
    df_sc_bin_avg["corr_0"] = df_sc["corr_0"].abs()
    df_sc_bin_avg["corr_max"] = df_sc["corr_max"]
    df_sc_bin_sem["corr_0_sem"] = df_sc["corr_0"].abs()
    df_sc_bin_sem["corr_max_sem"] = df_sc["corr_max"]

    # Reset bins index and compute sem
    df_sc_bin_avg = df_sc_bin_avg.groupby("dist", observed=False).mean(
        numeric_only=True
    )
    df_sc_bin_avg["bin"] = bins_values[:-1] + np.diff(bins_values) / 2
    df_sc_bin_avg.reset_index(inplace=True, drop=True)
    df_sc_bin_sem = df_sc_bin_sem.groupby("dist", observed=False).sem(numeric_only=True)
    df_sc_bin_sem.reset_index(inplace=True, drop=True)
    df_sc_bin = pd.concat([df_sc_bin_avg, df_sc_bin_sem], axis=1)
    df_sc_bin.dropna(inplace=True)

    return df_sc_bin


def _exp_decay(x, k, a, b):
    return a * np.exp(-x / k) + b


def fit_sc(df_sc: pd.DataFrame, col_name="corr_max", upper_bounds=(100, 1, 1)):

    x = df_sc["dist"].to_numpy(dtype=float)
    y = df_sc[col_name].to_numpy(dtype=float)
    if len(y) < 20:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    try:
        popt, pcov = optimize.curve_fit(
            _exp_decay,
            x,
            y,
            bounds=((0, 0, 0), upper_bounds),
        )
    except RuntimeError:
        popt = np.array([np.nan, np.nan, np.nan])
        pcov = np.nan

    return popt, pcov


def fit_sc_bins(df_sc_bin: pd.DataFrame, col_name="corr_max", upper_bounds=(100, 1, 1)):

    x = df_sc_bin["bin"].to_numpy(dtype=float)
    y = df_sc_bin[col_name].to_numpy(dtype=float)
    y_err = df_sc_bin[col_name + "_sem"].to_numpy(dtype=float)
    if len(y) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    try:
        popt, pcov = optimize.curve_fit(
            _exp_decay,
            x,
            y,
            sigma=y_err,
            bounds=((0, 0, 0), upper_bounds),
        )
    except RuntimeError:
        popt = np.array([np.nan, np.nan, np.nan])
        pcov = np.nan

    return popt, pcov


###
# Generate null models
# The following methods are imported from the netneurotools package (https://github.com/netneurolab/netneurotools)
###


def _gen_rotation(seed=None):
    """
    Generate random matrix for rotating spherical coordinates.

    Parameters
    ----------
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation

    Returns
    -------
    rotate_{l,r} : (3, 3) numpy.ndarray
        Rotations for left and right hemisphere coordinates, respectively
    """
    rs = check_random_state(seed)

    # for reflecting across Y-Z plane
    reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # generate rotation for left
    rotate_l, temp = np.linalg.qr(rs.normal(size=(3, 3)))
    rotate_l = rotate_l @ np.diag(np.sign(np.diag(temp)))
    if np.linalg.det(rotate_l) < 0:
        rotate_l[:, 0] = -rotate_l[:, 0]

    # reflect the left rotation across Y-Z plane
    rotate_r = reflect @ rotate_l @ reflect

    return rotate_l, rotate_r


def gen_spinsamples(
    coords,
    hemiid,
    n_rotate=1000,
    check_duplicates=True,
    method="original",
    exact=False,
    seed=None,
    verbose=False,
    return_cost=False,
):
    """
    Return a resampling array for `coords` obtained from rotations / spins.

    Using the method initially proposed in [ST1]_ (and later modified + updated
    based on findings in [ST2]_ and [ST3]_), this function applies random
    rotations to the user-supplied `coords` in order to generate a resampling
    array that preserves its spatial embedding. Rotations are generated for one
    hemisphere and mirrored for the other (see `hemiid` for more information).

    Due to irregular sampling of `coords` and the randomness of the rotations
    it is possible that some "rotations" may resample with replacement (i.e.,
    will not be a true permutation). The likelihood of this can be reduced by
    either increasing the sampling density of `coords` or changing the
    ``method`` parameter (see Notes for more information on the latter).

    Parameters
    ----------
    coords : (N, 3) array_like
        X, Y, Z coordinates of `N` nodes/parcels/regions/vertices defined on a
        sphere
    hemiid : (N,) array_like
        Array denoting hemisphere designation of coordinates in `coords`, where
        values should be {0, 1} denoting the different hemispheres. Rotations
        are generated for one hemisphere and mirrored across the y-axis for the
        other hemisphere.
    n_rotate : int, optional
        Number of rotations to generate. Default: 1000
    check_duplicates : bool, optional
        Whether to check for and attempt to avoid duplicate resamplings. A
        warnings will be raised if duplicates cannot be avoided. Setting to
        True may increase the runtime of this function! Default: True
    method : {'original', 'vasa', 'hungarian'}, optional
        Method by which to match non- and rotated coordinates. Specifying
        'original' will use the method described in [ST1]_. Specfying 'vasa'
        will use the method described in [ST4]_. Specfying 'hungarian' will use
        the Hungarian algorithm to minimize the global cost of reassignment
        (will dramatically increase runtime). Default: 'original'
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Default: None
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True

    Returns
    -------
    spinsamples : (N, `n_rotate`) numpy.ndarray
        Resampling matrix to use in permuting data based on supplied `coords`.
    cost : (N, `n_rotate`,) numpy.ndarray
        Cost (specified as Euclidean distance) of re-assigning each coordinate
        for every rotation in `spinsamples`. Only provided if `return_cost` is
        True.

    Notes
    -----
    By default, this function uses the minimum Euclidean distance between the
    original coordinates and the new, rotated coordinates to generate a
    resampling array after each spin. Unfortunately, this can (with some
    frequency) lead to multiple coordinates being re-assigned the same value:

        >>> from netneurotools import stats as nnstats
        >>> coords = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
        >>> hemi = [0, 0, 1, 1]
        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='original', check_duplicates=False)
        array([[0],
               [0],
               [2],
               [3]])

    While this is reasonable in most circumstances, if you feel incredibly
    strongly about having a perfect "permutation" (i.e., all indices appear
    once and exactly once in the resampling), you can set the ``method``
    parameter to either 'vasa' or 'hungarian':

        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='vasa', check_duplicates=False)
        array([[1],
               [0],
               [2],
               [3]])
        >>> nnstats.gen_spinsamples(coords, hemi, n_rotate=1, seed=1,
        ...                         method='hungarian', check_duplicates=False)
        array([[0],
               [1],
               [2],
               [3]])

    Note that setting this parameter may increase the runtime of the function
    (especially for `method='hungarian'`). Refer to [ST1]_ for information on
    why the default (i.e., ``exact`` set to False) suffices in most cases.

    For the original MATLAB implementation of this function refer to [ST5]_.

    References
    ----------
    .. [ST1] Alexander-Bloch, A., Shou, H., Liu, S., Satterthwaite, T. D.,
       Glahn, D. C., Shinohara, R. T., Vandekar, S. N., & Raznahan, A. (2018).
       On testing for spatial correspondence between maps of human brain
       structure and function. NeuroImage, 178, 540-51.

    .. [ST2] Blaser, R., & Fryzlewicz, P. (2016). Random Rotation Ensembles.
       Journal of Machine Learning Research, 17(4), 1–26.

    .. [ST3] Lefèvre, J., Pepe, A., Muscato, J., De Guio, F., Girard, N.,
       Auzias, G., & Germanaud, D. (2018). SPANOL (SPectral ANalysis of Lobes):
       A Spectral Clustering Framework for Individual and Group Parcellation of
       Cortical Surfaces in Lobes. Frontiers in Neuroscience, 12, 354.

    .. [ST4] Váša, F., Seidlitz, J., Romero-Garcia, R., Whitaker, K. J.,
       Rosenthal, G., Vértes, P. E., ... & Jones, P. B. (2018). Adolescent
       tuning of association cortex in human structural brain networks.
       Cerebral Cortex, 28(1), 281-294.

    .. [ST5] https://github.com/spin-test/spin-test
    """
    methods = ["original", "vasa", "hungarian"]
    if method not in methods:
        raise ValueError(
            'Provided method "{}" invalid. Must be one of {}.'.format(method, methods)
        )

    if exact:
        warnings.warn(
            "The `exact` parameter will no longer be supported in "
            "an upcoming release. Please use the `method` parameter "
            "instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if exact == "vasa" and method == "original":
            method = "vasa"
        elif exact and method == "original":
            method = "hungarian"

    seed = check_random_state(seed)

    coords = np.asanyarray(coords)
    hemiid = np.squeeze(np.asanyarray(hemiid, dtype="int8"))

    # check supplied coordinate shape
    if coords.shape[-1] != 3 or coords.squeeze().ndim != 2:
        raise ValueError(
            "Provided `coords` must be of shape (N, 3), not {}".format(coords.shape)
        )

    # ensure hemisphere designation array is correct
    if hemiid.ndim != 1:
        raise ValueError("Provided `hemiid` array must be one-dimensional.")
    if len(coords) != len(hemiid):
        raise ValueError(
            "Provided `coords` and `hemiid` must have the same "
            "length. Provided lengths: coords = {}, hemiid = {}".format(
                len(coords), len(hemiid)
            )
        )
    if np.max(hemiid) > 1 or np.min(hemiid) < 0:
        raise ValueError(
            "Hemiid must have values in {0, 1} denoting left and "
            "right hemisphere coordinates, respectively. "
            + "Provided array contains values: {}".format(np.unique(hemiid))
        )

    # empty array to store resampling indices
    spinsamples = np.zeros((len(coords), n_rotate), dtype=int)
    cost = np.zeros((len(coords), n_rotate))
    inds = np.arange(len(coords), dtype=int)

    # generate rotations and resampling array!
    msg, warned = "", False
    for n in range(n_rotate):
        count, duplicated = 0, True

        if verbose:
            msg = "Generating spin {:>5} of {:>5}".format(n, n_rotate)
            print(msg, end="\r", flush=True)

        while duplicated and count < 500:
            count, duplicated = count + 1, False
            resampled = np.zeros(len(coords), dtype="int32")

            # rotate each hemisphere separately
            for h, rot in enumerate(_gen_rotation(seed=seed)):
                hinds = hemiid == h
                coor = coords[hinds]
                if len(coor) == 0:
                    continue

                # if we need an "exact" mapping (i.e., each node needs to be
                # assigned EXACTLY once) then we have to calculate the full
                # distance matrix which is a nightmare with respect to memory
                # for anything that isn't parcellated data.
                # that is, don't do this with vertex coordinates!
                if method == "vasa":
                    dist = spatial.distance_matrix(coor, coor @ rot)
                    # min of max a la Vasa et al., 2018
                    col = np.zeros(len(coor), dtype="int32")
                    for _ in range(len(dist)):
                        # find parcel whose closest neighbor is farthest away
                        # overall; assign to that
                        row = dist.min(axis=1).argmax()
                        col[row] = dist[row].argmin()
                        cost[inds[hinds][row], n] = dist[row, col[row]]
                        # set to -inf and inf so they can't be assigned again
                        dist[row] = -np.inf
                        dist[:, col[row]] = np.inf
                # optimization of total cost using Hungarian algorithm. this
                # may result in certain parcels having higher cost than with
                # `method='vasa'` but should always result in the total cost
                # being lower #tradeoffs
                elif method == "hungarian":
                    dist = spatial.distance_matrix(coor, coor @ rot)
                    row, col = optimize.linear_sum_assignment(dist)
                    cost[hinds, n] = dist[row, col]
                # if nodes can be assigned multiple targets, we can simply use
                # the absolute minimum of the distances (no optimization
                # required) which is _much_ lighter on memory
                # huge thanks to https://stackoverflow.com/a/47779290 for this
                # memory-efficient method
                elif method == "original":
                    dist, col = spatial.cKDTree(coor @ rot).query(coor, 1)
                    cost[hinds, n] = dist

                resampled[hinds] = inds[hinds][col]

            # if we want to check for duplicates ensure that we don't have any
            if check_duplicates:
                if np.any(np.all(resampled[:, None] == spinsamples[:, :n], 0)):
                    duplicated = True
                # if our "spin" is identical to the input then that's no good
                elif np.all(resampled == inds):
                    duplicated = True

        # if we broke out because we tried 500 rotations and couldn't generate
        # a new one, warn that we're using duplicate rotations and give up.
        # this should only be triggered if check_duplicates is set to True
        if count == 500 and not warned:
            warnings.warn(
                "Duplicate rotations used. Check resampling array "
                "to determine real number of unique permutations.",
                stacklevel=2,
            )
            warned = True

        spinsamples[:, n] = resampled

    if verbose:
        print(" " * len(msg) + "\b" * len(msg), end="", flush=True)

    if return_cost:
        return spinsamples, cost

    return spinsamples


def get_rho_boot(x: np.ndarray, y: np.ndarray, corr_type="spearman", nboot=9999):

    # Select function to use
    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "pearson":
        corr_func = pearsonr

    # Define function for bootstrapped statistic
    def my_stat(s1, s2):
        return corr_func(s1, s2)[0]

    # Compute bootstraps
    res = bootstrap(
        (x, y),
        my_stat,
        n_resamples=nboot,
        vectorized=False,
        paired=True,
        random_state=290496,
    )

    return res


def get_pcorr(
    x: np.ndarray,
    y: np.ndarray,
    map_coords: np.ndarray,
    hemiid=None,
    nspins=1000,
    method="original",
    corr_type="spearman",
):
    """_summary_

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        map_coords (np.ndarray): _description_
        hemiid (_type_, optional): _description_. Defaults to None.
        nspins (int, optional): _description_. Defaults to 1000.
        method (str, optional): _description_. Defaults to "original".
        corr_type (str, optional): _description_. Defaults to "spearman".

    Returns:
        _type_: _description_
    """

    # Check hemiid
    if hemiid is None:
        # Assume map is symmetric around half
        n = map_coords.shape[0] // 2
        hemiid = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    # Generate permuted maps
    print(f"Generating {nspins} permutations...")
    spins = gen_spinsamples(map_coords, hemiid, nspins, method, seed=290496)

    # Compute correlation on real data
    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "pearson":
        corr_func = pearsonr
    rho, _ = corr_func(x, y)

    # Compute correlation on permuted data
    permuted_p = np.zeros(nspins)
    # Create "copies" of x and y for the two hemispheres
    xx = np.concatenate([x, x])
    yy = np.concatenate([y, y])
    for spin in range(nspins):
        permuted_p[spin] = corr_func(xx[spins[:, spin]], yy)[0]

    # Compute p-value
    permmean = np.mean(permuted_p)
    p_corr = (abs(permuted_p - permmean) > abs(rho - permmean)).mean()

    return rho, p_corr


def get_pcorr_mnia(
    x: pd.DataFrame,
    y: pd.DataFrame,
    map_coords: pd.DataFrame,
    hemiid=None,
    nspins=1000,
    method="original",
    corr_type="spearman",
):

    # Separate cortical and amygdala/hippocampus regions
    x_ctx = x.drop(["Amygdala", "Hippocampus"])
    x_ha = x.loc[["Amygdala", "Hippocampus"]].copy()
    y_ctx = y.drop(["Amygdala", "Hippocampus"])
    y_ha = y.loc[["Amygdala", "Hippocampus"]].copy()

    # Make sure that coordinates have the same index as the data
    idx_order = [i[:-3] for i in map_coords.index[: len(map_coords) // 2]]
    x_ctx = x_ctx.loc[idx_order]
    y_ctx = y_ctx.loc[idx_order]

    # Check hemiid
    if hemiid is None:
        # Assume map is symmetric around half
        n = map_coords.shape[0] // 2
        hemiid = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    # Generate permuted maps
    print(f"Generating {nspins} permutations...")
    spins = gen_spinsamples(map_coords.to_numpy(), hemiid, nspins, method, seed=290496)

    # Compute correlation on real data
    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "pearson":
        corr_func = pearsonr
    rho, _ = corr_func(x, y)

    # Compute correlation on permuted data
    permuted_p = np.zeros(nspins)
    # Create "copies" of x and y for the two hemispheres
    xx = pd.concat([x_ctx, x_ctx])
    yy = pd.concat([y_ctx, y_ctx])
    for spin in range(nspins):
        # Re-append hip / amy electrodes for computation
        x_perm = xx.iloc[spins[:, spin]].to_list() + x_ha.to_list() * 2
        y_perm = yy.to_list() + y_ha.to_list() * 2
        permuted_p[spin] = corr_func(x_perm, y_perm)[0]

    # Compute p-value
    permmean = np.mean(permuted_p)
    p_corr = (abs(permuted_p - permmean) > abs(rho - permmean)).mean()

    return rho, p_corr
