from scipy import signal, optimize
import numpy as np
import pandas as pd
from pybispectra import compute_fft, TDE


def _check_hemi(mni_x: float):
    if mni_x < 0:
        return "left"
    else:
        return "right"


def _get_mni(df_info: pd.DataFrame, chan: str):

    return df_info.loc[df_info["chan"] == chan, ["mni_x", "mni_y", "mni_z"]].to_numpy(
        dtype=float
    )


def _compute_cc(data_ch1: np.ndarray, data_ch2: np.ndarray):

    data_ch1 = data_ch1.squeeze()
    data_ch2 = data_ch2.squeeze()
    # Loop over epochs
    corr_avg = []
    for k in range(len(data_ch1)):
        ch1_norm = (data_ch1[k] - data_ch1[k].mean()) / np.linalg.norm(data_ch1[k])
        ch2_norm = (data_ch2[k] - data_ch2[k].mean()) / np.linalg.norm(data_ch2[k])
        corr = signal.correlate(ch1_norm, ch2_norm, mode="same")
        lags = signal.correlation_lags(len(ch1_norm), len(ch2_norm), mode="same")
        corr_avg.append(corr)
    corr_avg = np.array(corr_avg).mean(axis=0)
    corr_max = np.max(np.abs(corr_avg))
    lag_max = lags[np.argmax(np.abs(corr_avg))]

    return corr_max, lag_max


def _compute_cc_bispectrum(
    data_ch1: np.ndarray, data_ch2: np.ndarray, sfreq: int, freq_band=None
):

    # Compute and use bispectrum
    fft_coeffs, freqs = compute_fft(
        data=np.concatenate([data_ch1, data_ch2], axis=1),
        sampling_freq=float(sfreq),
        n_points=2 * data_ch1.shape[-1] + 1,  # recommended for time delay estimation
        window="hamming",
        verbose=False,
    )
    # Initialise TDE object
    tde = TDE(
        data=fft_coeffs,
        freqs=freqs,
        sampling_freq=float(sfreq),
        verbose=False,
    )
    if freq_band is None:
        band_freqs = (0, sfreq / 2)
    # Compute TDE
    tde.compute(
        indices=((0,), (1,)),
        fmin=(band_freqs[0]),
        fmax=(band_freqs[1]),
        method=1,
        antisym=True,
    )
    tde_times = tde.results.times
    tde_results = tde.results.get_results()  # return results as array
    # Get max correlation and lag
    corr_max = np.max(tde_results)
    lag_max = tde_times[np.argmax(tde_results)]

    return corr_max, lag_max


def compute_cc(
    epochs,
    df_info: pd.DataFrame,
    df_regions: pd.DataFrame,
    use_bispectrum=False,
) -> pd.DataFrame:
    """Compute cross-correlation (cc) between every pair of channels.

    Args:
        epochs (mne.Epochs): epochs object with data
        df_info (pd.DataFrame): dataframe with information on channels
        df_regions (pd.DataFrame): map from region names to lobes
        use_bispectrum (bool, optional): use bispectrum estimation of cross-correlation. Defaults to False.

    Returns:
        pd.DataFrame: SC values (max lag and correlation) for every pair of channels.
    """

    # Lists for results
    ch1_list = []
    lobe1_list = []
    reg1_list = []
    ch2_list = []
    lobe2_list = []
    reg2_list = []
    dist = []
    corr_max = []
    lag_max = []

    # Loop on the "seed" channel
    for i, ch1 in enumerate(epochs.ch_names):
        mni1 = _get_mni(df_info, ch1)
        # Loop on the "target" channel
        for j in range(i + 1, len(epochs.ch_names)):
            ch2 = epochs.ch_names[j]
            # Get MNI coordinates
            mni2 = _get_mni(df_info, ch2)
            if ch1 == ch2:
                continue
            # Retain only pairs on the same hemisphere
            hemi1, hemi2 = _check_hemi(mni1[0]), _check_hemi(mni2[0])
            if hemi1 != hemi2:
                continue

            # Register info on electrodes location
            ch1_list.append(ch1)
            reg1_list.append(df_info.loc[df_info["chan"] == ch1, "region"])
            lobe1_list.append(
                df_regions[df_regions["Region name"] == reg1_list[-1]]["Lobe"].values[0]
            )
            ch2_list.append(ch2)
            reg2_list.append(df_info.loc[df_info["chan"] == ch2, "region"])
            lobe2_list.append(
                df_regions[df_regions["Region name"] == reg2_list[-1]]["Lobe"].values[0]
            )
            dist.append(np.linalg.norm(mni1 - mni2))

            # Extract data
            data_ch1 = epochs.get_data(picks=ch1)
            data_ch2 = epochs.get_data(picks=ch2)

            # Use "classic" cross-correlation or bispectrum
            if not use_bispectrum:
                corr_max_pair, lag_max_pair = _compute_cc(data_ch1, data_ch2)

            else:
                corr_max_pair, lag_max_pair = _compute_cc_bispectrum(
                    data_ch1,
                    data_ch2,
                    sfreq=epochs.info["sfreq"],
                    freq_band=(epochs.info["lowpass"], epochs.info["highpass"]),
                )

            corr_max.append(corr_max_pair)
            lag_max.append(lag_max_pair)

    df_sc = pd.DataFrame(
        columns=[
            "ch_1",
            "ch_2",
            "lobe_1",
            "lobe_2",
            "region_1",
            "region_2",
            "dist",
            "corr",
            "lag",
        ]
    )
    df_sc["ch_1"] = ch1_list
    df_sc["ch_2"] = ch2_list
    df_sc["region_1"] = reg1_list
    df_sc["region_2"] = reg2_list
    df_sc["lobe_1"] = lobe1_list
    df_sc["lobe_2"] = lobe2_list
    df_sc["dist"] = dist
    df_sc["corr"] = corr_max
    df_sc["lag"] = lag_max

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
    df_sc_bin_avg = pd.DataFrame(columns=["dist", "bin", "corr", "lag"])
    df_sc_bin_sem = pd.DataFrame(columns=["dist", "corr_sem", "lag_sem"])
    df_sc_bin_avg["dist"] = bins_cat
    df_sc_bin_sem["dist"] = bins_cat

    # Compute average and sem for each bin
    df_sc_bin_avg["corr"] = df_sc["corr"]
    df_sc_bin_avg["lag"] = df_sc["lag"]
    df_sc_bin_sem["corr_sem"] = df_sc["corr"]
    df_sc_bin_sem["lag_sem"] = df_sc["lag"]

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


def fit_sc(df_sc: pd.DataFrame, col_name: str, n_min=20, upper_bounds=(100, 1, 1)):

    x = df_sc["dist"].to_numpy(dtype=float)
    y = df_sc[col_name].to_numpy(dtype=float)
    if len(y) < n_min:
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


def fit_sc_bins(
    df_sc_bin: pd.DataFrame, col_name: str, n_min=2, upper_bounds=(100, 1, 1)
):

    x = df_sc_bin["bin"].to_numpy(dtype=float)
    y = df_sc_bin[col_name].to_numpy(dtype=float)
    y_err = df_sc_bin[col_name + "_sem"].to_numpy(dtype=float)
    if len(y) < n_min:
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
