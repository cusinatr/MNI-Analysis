from pathlib import Path
import warnings
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy import signal
from scipy import optimize, spatial
from scipy.stats import pearsonr, spearmanr, bootstrap
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


def create_epo(raw: mne.io.RawArray, freq_band=False, band_freqs=[]) -> mne.Epochs:
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
    df_sc_bin_avg = df_sc_bin_avg.groupby(
        "dist", observed=False
    ).mean(numeric_only=True)
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
