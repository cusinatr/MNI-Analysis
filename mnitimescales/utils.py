from pathlib import Path
import warnings
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy import signal
from scipy import optimize, spatial
from scipy.stats import pearsonr, spearmanr, bootstrap, false_discovery_control
import statsmodels.formula.api as smf
from sklearn.utils.validation import check_random_state


###
# General functions
###


def create_res_df(
    df_info_pat: pd.DataFrame,
    chs_good: list,
    stage: str,
    columns_res=[],
):
    """Create a dataframe for results of a patient.

    Args:
        df_info_pat (pd.DataFrame): metadata about patient channels.
        pat_id (str): ID for patient.
        age (float): age of patient.
        gender (str): gender of patient.
        stage (str): sleep stage.
        columns_res (list): column names to add for results.

    Returns:
        pd.DataFrame: dataframe with metadata for patient and empty results column(s).
    """

    df_res_pat = pd.DataFrame(
        columns=[
            "pat",
            "age",
            "gender",
            "chan",
            "type",
            "stage",
            "region",
            "mni_x",
            "mni_y",
            "mni_z",
        ]
        + columns_res
    )
    df_info_pat_chs = df_info_pat[df_info_pat["chan"].isin(chs_good)]
    df_res_pat["pat"] = df_info_pat_chs["pat"].to_list()
    df_res_pat["age"] = df_info_pat_chs["age"].to_list()
    df_res_pat["gender"] = df_info_pat_chs["gender"].to_list()
    df_res_pat["chan"] = df_info_pat_chs["chan"].to_list()
    df_res_pat["type"] = df_info_pat_chs["type"].to_list()
    df_res_pat["stage"] = [stage] * len(df_info_pat_chs)
    df_res_pat["region"] = df_info_pat_chs["region"].to_list()
    df_res_pat["mni_x"] = df_info_pat_chs["mni_x"].to_list()
    df_res_pat["mni_y"] = df_info_pat_chs["mni_y"].to_list()
    df_res_pat["mni_z"] = df_info_pat_chs["mni_z"].to_list()

    return df_res_pat


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
# Functions for slow waves and timescales
###


def divide_sws(
    data: dict,
    epo_stages: np.ndarray,
    epo_times: np.ndarray,
    epo_dur: float,
    data_type: str,
    sw_events: pd.DataFrame,
    sw_overlap: dict,
) -> dict:
    """Divide epochs in SWs and non-SWs periods.

    Args:
        data (dict): ACF/PSD data for each epoch and channel.
        epo_stages (np.ndarray): shape (n_epo,), stage for each epoch.
        epo_times (np.ndarray): shape (n_epo,), starting time for each epoch.
        epo_dur (float): duration of epochs in s.
        data_type (str): either "acf" or "psd".
        sw_events (pd.DataFrame): dataframe with SW events.
        sw_overlap (dict): overlap matrices for each channel. Keys are channel names.

    Returns:
        dict: keys are conditions and lags/freqs, values are data for each condition.
    """

    # Choose based on data type
    if data_type == "acf":
        x_name = "lags"
    elif data_type == "psd":
        x_name = "freqs"

    # Check stages of events
    stages = np.unique(sw_events.Stage)

    # Restrict epochs to stages with SWs
    idx_stages = np.where(np.isin(epo_stages, stages))[0]
    data_stages = data[data_type][idx_stages, :, :]

    # Get good channels
    all_chs = data["chans"].copy()
    good_chs = [ch for ch in all_chs if ch in list(sw_overlap.keys())]
    idx_good_chs = [i for i, ch in enumerate(all_chs) if ch in good_chs]

    # Initialize data container
    data_sws = {
        "sws": {ch: [] for ch in good_chs},
        "sws_loc": {ch: [] for ch in good_chs},
        "sws_glo": {ch: [] for ch in good_chs},
        "nsws": {ch: [] for ch in good_chs},
    }

    # Create a 2D array of epoch start times
    epoch_starts = epo_times[idx_stages].reshape(-1, 1)

    # Compute threshold for global waves
    glo_thre = _compute_sw_global_threshold(sw_overlap)

    # Loop over channels separately
    for i, ch in tqdm(enumerate(good_chs), total=len(good_chs)):
        # Get SWs and overlaps for channel
        sw_events_ch = sw_events[sw_events.Channel == ch]
        sw_overlap_ch = sw_overlap[ch]

        # Check for overlapping SWs using broadcasting
        sw_starts = sw_events_ch.Start.to_numpy().reshape(1, -1)
        sw_ends = sw_events_ch.End.to_numpy().reshape(1, -1)
        mask_overlap_epo = (
            (sw_starts >= epoch_starts) & (sw_starts < epoch_starts + epo_dur)
        ) | ((sw_ends >= epoch_starts) & (sw_ends < epoch_starts + epo_dur))

        # Check for at least SWs in epochs and local / global SWs
        mask_sws = np.any(mask_overlap_epo, axis=1)
        idx_loc = np.where(sw_overlap_ch.mean(axis=1) < glo_thre)[0]
        idx_glo = np.where(sw_overlap_ch.mean(axis=1) >= glo_thre)[0]
        # Create masks, making sure there are no duplicates
        mask_sws_glo = np.any(mask_overlap_epo[:, idx_glo], axis=1)
        mask_sws_loc = np.logical_and(
            np.any(mask_overlap_epo[:, idx_loc], axis=1), np.logical_not(mask_sws_glo)
        )

        # Add to sws or nsws
        idx_ch = idx_good_chs[i]
        data_sws["sws"][ch] = data_stages[mask_sws, idx_ch]
        data_sws["sws_loc"][ch] = data_stages[mask_sws_loc, idx_ch]
        data_sws["sws_glo"][ch] = data_stages[mask_sws_glo, idx_ch]
        data_sws["nsws"][ch] = data_stages[~mask_sws, idx_ch]

    # Add xvalues back
    data_sws[x_name] = data[x_name].copy()

    return data_sws


###
# Spatial analysis
###

def get_tc_sc_corr(df_sc: dict, df_timescales: pd.DataFrame, stages: list, distances: np.ndarray, 
map_coords: np.ndarray, corr_type="pearson") -> dict:

    # Compute dataframe for average cross-correlation per distance block
    index = pd.MultiIndex.from_product(
        [list(df_sc[stages[0]]["region_1"].unique()), distances],
        names=["region", "distance"],
    )
    df_avg_d = pd.DataFrame(
        columns=stages,
        index=index,
    )
    delta_d = distances[1] - distances[0]  # Delta of distances blocks
    for reg in df_avg_d.index.get_level_values("region").unique():
        for dist in distances:
            for stage in stages:
                df_avg_d.loc[(reg, dist), stage] = (
                    df_sc[stage]["corr"][
                        (
                            (df_sc[stage]["region_1"] == reg)
                            | (df_sc[stage]["region_2"] == reg)
                        )
                        & (df_sc[stage]["dist"] <= dist + delta_d)
                        & (df_sc[stage]["dist"] > dist - delta_d)
                    ]
                    .abs()
                    .mean()
                )
    df_avg_d.reset_index(inplace=True)

    # Compute correlations with TC per stage
    df_rhos_d = {}
    for stage in stages:
        df_rhos_d_stage = pd.DataFrame(index=df_avg_d["distance"].unique(), columns=["rho", "rho_se", "pval"], dtype=float)
        # Get average tau per MNI region
        df_tau_stage_mni = get_avg_tau_mni(
            df_timescales[df_timescales["stage"] == stage].copy(), method="LME"
        )
        for dist in df_avg_d["distance"].unique():
            # Get spatial parameter
            df_spa = df_avg_d[df_avg_d["distance"] == dist].set_index("region")
            df_spa = df_spa[stage].dropna()
            # Map coords
            map_coords_dist = map_coords.loc[
                list(df_spa.index.drop(["Amygdala", "Hippocampus"], errors="ignore") + "_lh")
                + list(df_spa.index.drop(["Amygdala", "Hippocampus"], errors="ignore") + "_rh")
            ]
            # Get correlation values
            rho, p_corr = get_pcorr_mnia(
                df_tau_stage_mni.loc[df_spa.index],
                df_spa,
                map_coords_dist,
                method="vasa",
                corr_type=corr_type,
            )
            rho_boot = get_rho_boot(
                df_tau_stage_mni.loc[df_spa.index],
                df_spa, corr_type=corr_type, nboot=1000  # keep bootstraps lower for comp. time
            )
            df_rhos_d_stage.loc[dist, "rho"] = rho
            df_rhos_d_stage.loc[dist, "rho_se"] = rho_boot.standard_error
            df_rhos_d_stage.loc[dist, "pval"] = p_corr

        # Correct p-values with FDR
        df_rhos_d_stage.loc[:, "pval"] = false_discovery_control(df_rhos_d_stage.loc[:, "pval"])

        df_rhos_d[stage] = df_rhos_d_stage.copy()

    return df_rhos_d

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
    ha_keys = y.index.intersection(["Amygdala", "Hippocampus"])
    x_ctx = x.drop(ha_keys)
    x_ha = x.loc[ha_keys].copy()
    y_ctx = y.drop(ha_keys)
    y_ha = y.loc[ha_keys].copy()

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
        x_perm = xx.iloc[spins[:, spin]].to_list() + x_ha.to_list() * len(ha_keys)
        y_perm = yy.to_list() + y_ha.to_list() * len(ha_keys)
        permuted_p[spin] = corr_func(x_perm, y_perm)[0]

    # Compute p-value
    permmean = np.mean(permuted_p)
    p_corr = (abs(permuted_p - permmean) > abs(rho - permmean)).mean()

    return rho, p_corr
