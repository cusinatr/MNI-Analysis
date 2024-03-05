from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import mne
import nibabel as nib
import statistics
import statsmodels.formula.api as smf


###
# General functions
###


def get_avg_tau_mni(data: pd.DataFrame):

    # Remove NaN if present
    data = data.dropna()

    # Mixed model
    md = smf.mixedlm("tau ~ 0 + region", data, groups=data["pat"])
    mdf = md.fit()

    # Extract fitted parameters
    data_mni = mdf.fe_params
    data_mni.index = data_mni.index.str.strip("region[").str.strip("]")

    return data_mni


###
# Functions for creating Raw and Epochs
###


def create_RawMNE(data: np.ndarray, chans: list, sfreq: int, return_all=False):
    """Create MNE Raw object from array data.

    Args:
        data (np.ndarray): (n_chs, n_times) array of data
        chans (list): chennel names
        sfreq (int): sampling frequency
        return_all (bool, optional): Whether to return also indices of channels. Defaults to False.

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
        raw = mne.io.RawArray(data[idx_good], info, verbose=False)

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

        if return_all:
            return raw, idx_good, idx_nan
        return raw
    else:
        if return_all:
            return None, None, None
        return None


def create_epo(raw: mne.io.RawArray, freq_band: bool, band_freqs: list) -> mne.Epochs:
    """Create epochs from raw, additionally filtering in a band.

    Args:
        raw (mne.io.RawArray): Raw object
        freq_band (bool): Whether to filter in a freq band and get the amplitude
        band_freqs (list): Limits of the frequency band

    Returns:
        mne.Epochs: "surrogate" epochs
    """

    if freq_band:
        # Get band amplitude
        raw.filter(band_freqs[0], band_freqs[1], verbose=False)
        raw.apply_hilbert(envelope=True)
        # Apply log-transformation to make data more "normal"
        raw._data = np.log(raw._data**2)

    # Create epochs to discard flat segments
    epo = mne.make_fixed_length_epochs(
        raw, duration=1, overlap=0.5, preload=True, verbose=False
    )

    return epo


###
# Functions for parcellated metric
###


def apply_affine(f_affine, coord, forward):
    """
    Apply forward (index to MNI coor) or reverse (MNI to index) affine transformation.
    """
    M_aff, M_trsl = f_affine[:3, :3], f_affine[:3, -1]

    if forward:
        # index to MNI coordinate
        return np.dot(M_aff, coord) + M_trsl
    else:
        # MNI coordinate to index
        return np.dot(np.linalg.inv(M_aff), coord - M_trsl)


def _check_hemi(mni_x: float):
    if mni_x < 0:
        return "left"
    else:
        return "right"


def mni_to_mmp(mni_coord: np.ndarray, path_mmp: Path) -> list:

    # Load MMP data
    MMP_data = nib.load(path_mmp)
    MMP_map = MMP_data.get_fdata()
    MMP_map_nonzero = np.array(np.where(MMP_map > 0)).T
    MMP_map_nonzero_flat = MMP_map[np.where(MMP_map > 0)]

    # Retrieve annotations files
    subjects_dir = str(mne.datasets.sample.data_path()) + "/subjects"
    parc_map_name = "HCPMMP1"
    annot_file = subjects_dir + "/fsaverage/label/lh." + parc_map_name + ".annot"
    _, _, parc_names = nib.freesurfer.read_annot(annot_file)
    region_names = [n[2:-4].decode("utf-8") for n in parc_names]

    # Get closest parcel
    closest_labs = []
    for coord in mni_coord:
        mni_coord_affine = apply_affine(MMP_data.affine, coord, False)
        vertex_dists = np.linalg.norm(
            MMP_map_nonzero - mni_coord_affine.reshape(1, -1), axis=1
        )
        idx_lab = statistics.mode(
            MMP_map_nonzero_flat[np.argsort(vertex_dists)[:5]]
        )  # mode of 5 closest
        idx_lab = int(idx_lab)
        closest_labs.append(region_names[idx_lab])

    return closest_labs


def compute_parc_metric(
    df_patients: pd.DataFrame,
    output_grid: np.ndarray,
    affine_transf: np.ndarray,
    d: float,
    feature="tau",
):
    # Smoothing parameter: Gaussian is at 50% when d voxels away
    d_alpha = d / (-np.log(0.5)) ** 0.5

    feat_weighted, W_max = [], []
    for i_p in df_patients["pat"].unique():
        # iterate over patients
        print(int(i_p), end="|")
        df_patient = df_patients[df_patients["pat"] == i_p]

        # get transformed ECoG coordinate
        input_grid = np.array(
            [
                apply_affine(
                    affine_transf, row[["mni_x", "mni_y", "mni_z"]].values, False
                )
                for _, row in df_patient.iterrows()
            ]
        )
        input_grid = input_grid.astype(np.float64)

        # create the weight matrix from input to output projection based on Gaussian weighting of Euclidean distance
        W_mat = np.zeros((input_grid.shape[0], output_grid.shape[0]))
        for ig in range(input_grid.shape[0]):
            W_mat[ig, :] = np.exp(
                -np.linalg.norm(output_grid - input_grid[ig, :], axis=1) ** 2
                / d_alpha**2
            )

        # Get total and max weights to drop bad coverage points
        W_max.append(np.max(W_mat, axis=0))
        feat_weighted.append(
            np.dot(df_patient[feature].values, W_mat) / W_mat.sum(axis=0)
        )

    return feat_weighted, W_max


def compute_weighted_average(df_feature, df_W, w_thresh=0.5, axis=0, method="weighted"):
    if method == "weighted":
        # method 1: weighted average of all parcels
        return (df_feature * df_W).sum(axis=axis) / df_W.sum(axis=axis)

    elif method == "thresh_weighted":
        # method 2: weighted average of suprathreshold parcels
        thresh_mat = df_W >= w_thresh
        return (df_feature * df_W)[thresh_mat].sum(axis=axis) / df_W[thresh_mat].sum(
            axis=axis
        )

    elif method == "thresh_mean":
        # method 3: simple average of suprathreshold parcels
        thresh_mat = df_W >= w_thresh
        return np.nanmean((df_feature * df_W)[thresh_mat], axis=axis)

    else:
        raise ValueError(f"Method {method} not implemented!")


def compute_parc_metric_weight(
    feat_weighted: list,
    W_max: list,
    parcels: np.ndarray,
    parcels_names: np.ndarray,
    pats: np.ndarray,
    method="weighted",
):
    # Compute summary quantities per parcel (per subject)
    # Max weight
    df_W_max = pd.DataFrame(np.array(W_max).T, columns=np.unique(pats).astype(int))
    df_W_max.insert(0, "parcel", parcels)
    df_W_max_avg = df_W_max.groupby("parcel").max().T
    df_W_max_avg.columns = parcels_names
    # Weighted feature
    df_feat_weighted = pd.DataFrame(
        np.array(feat_weighted).T, columns=np.unique(pats).astype(int)
    )
    df_feat_weighted.insert(0, "parcel", parcels)
    df_feat_weighted_avg = df_feat_weighted.groupby("parcel").mean().T
    df_feat_weighted_avg.columns = parcels_names

    # (Weighted) average across subject
    df_feat_parc = compute_weighted_average(
        df_feat_weighted_avg, df_W_max_avg, method=method
    )

    return df_W_max_avg, df_feat_weighted_avg, df_feat_parc


###
# Functions for Spatial Correlations
###


def compute_SC(
    data: np.ndarray,
    df_info: pd.DataFrame,
    df_regions: pd.DataFrame,
    sfreq: int,
    path_mmp: Path,
    freq_band=False,
    band_freqs=[40, 80],
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
            data_ch1 = epo[:, i]

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

                # Loop over epochs
                corr_avg = []
                data_ch2 = epo[:, j]
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
                corr_max.append(np.max(corr_avg))
                lag_max.append(lags[np.argmax(corr_avg)])

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
    df_sc_bin_avg = df_sc_bin_avg.groupby("dist").mean()
    df_sc_bin_avg["bin"] = bins_values[:-1] + np.diff(bins_values) / 2
    df_sc_bin_avg.reset_index(inplace=True, drop=True)
    df_sc_bin_sem = df_sc_bin_sem.groupby("dist").sem()
    df_sc_bin_sem.reset_index(inplace=True, drop=True)
    df_sc_bin = pd.concat([df_sc_bin_avg, df_sc_bin_sem], axis=1)
    df_sc_bin.dropna(inplace=True)

    return df_sc_bin


def _exp_decay(x, k, a, b):
    return a * np.exp(-x / k) + b


def fit_sc(df_sc: pd.DataFrame):

    x = df_sc["dist"].to_numpy(dtype=float)
    y = df_sc["corr_max"].to_numpy(dtype=float)
    if len(y) < 20:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    try:
        popt, pcov = curve_fit(
            _exp_decay,
            x,
            y,
            bounds=((0, 0, -1), (100, 1, 1)),
        )
    except RuntimeError:
        popt = np.array([np.nan, np.nan, np.nan])
        pcov = np.nan

    return popt, pcov


def fit_sc_bins(df_sc_bin: pd.DataFrame):

    x = df_sc_bin["bin"].to_numpy(dtype=float)
    y = df_sc_bin["corr_max"].to_numpy(dtype=float)
    y_err = df_sc_bin["corr_max_sem"].to_numpy(dtype=float)
    if len(y) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    try:
        popt, pcov = curve_fit(
            _exp_decay,
            x,
            y,
            sigma=y_err,
            bounds=((0, 0, -1), (100, 1, 1)),
        )
    except RuntimeError:
        popt = np.array([np.nan, np.nan, np.nan])
        pcov = np.nan

    return popt, pcov
