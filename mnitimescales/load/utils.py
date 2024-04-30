###
# Functions for creating Raw and Epochs
###

import numpy as np
import mne


def create_RawMNE(
    data: np.ndarray,
    chans: list,
    sfreq: int,
    return_all=False,
    freq_band=False,
    band_freqs=[],
):
    """Create MNE Raw object from array data.

    Args:
        data (np.ndarray): (n_chs, n_times) array of data
        chans (list): chennel names
        sfreq (int): sampling frequency
        return_all (bool, optional): Whether to return also indices of channels. Defaults to False.
        freq_band (bool): Whether to filter in a freq band and get the amplitude
        band_freqs (list): Limits of the frequency band

    Returns:
        raw: mne.io.RawArray
        idx_good: indices of good channels (optional)
        idx_nan: indices of nan channels (optional)
    """

    # Detect nan channels
    chans_nan = np.array(chans)[np.where(np.all(np.isnan(data), axis=1))[0]]
    chans_good = [ch for ch in chans if ch not in chans_nan]

    if chans_good:
        idx_good = [i for i, ch in enumerate(chans) if ch in chans_good]
        idx_nan = [i for i, ch in enumerate(chans) if ch in chans_nan]
        info = mne.create_info(chans_good, sfreq, ch_types="seeg", verbose=False)
        raw = mne.io.RawArray(data[idx_good] / 1e6, info, verbose=False)  # convert to V

        # Detect flat time segments
        idx_flat = np.where(np.all(raw._data == 0, axis=0))[0]
        if idx_flat.size > 0:
            flat_start = [idx_flat[0]]
            flat_end = []
            idx_flat_diff = np.where(np.diff(idx_flat) != 1)[0]
            if idx_flat_diff.size > 0:
                flat_start.extend(idx_flat[idx_flat_diff + 1])
                flat_end.extend(idx_flat[idx_flat_diff])
            if idx_flat[-1] not in flat_end:
                flat_end.append(idx_flat[-1])
            flat_annot = mne.Annotations(
                onset=flat_start / sfreq,
                duration=(np.array(flat_end) - np.array(flat_start)) / sfreq,
                description="bad",
            )
            raw.set_annotations(flat_annot, verbose=False)

        if freq_band:
            # Get band amplitude
            raw.filter(band_freqs[0], band_freqs[1], verbose=False)
            raw.apply_hilbert(envelope=True)
            # Apply log-transformation to make data more "normal"
            raw._data = np.log(raw._data**2)

        if return_all:
            return raw, idx_good, idx_nan
        return raw
    else:
        if return_all:
            return None, None, None
        return None


def create_epo(
    raw: mne.io.RawArray,
    epo_dur: float,
    epo_overlap: float,
) -> mne.Epochs:
    """Create epochs from raw, additionally filtering in a band.

    Args:
        raw (mne.io.RawArray): Raw object
        epo_dur (float): Duration of epochs in seconds
        epo_overlap (float): Overlap of epochs in seconds

    Returns:
        mne.Epochs: "surrogate" epochs
    """

    # Create epochs to discard flat segments
    epo = mne.make_fixed_length_epochs(
        raw, duration=epo_dur, overlap=epo_overlap, preload=True, verbose=False
    )

    return epo
