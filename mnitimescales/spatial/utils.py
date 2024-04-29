


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