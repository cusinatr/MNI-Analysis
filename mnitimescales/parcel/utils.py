"""
Functions for parcellation.
The functions are mainly taken from: https://github.com/rdgao/field-echos/blob/master/echo_utils.py
Gao, R., van den Brink, R. L., Pfeffer, T., & Voytek, B. (2020). Neuronal timescales are
functionally dynamic and shaped by cortical microarchitecture. eLife, 9, e61277.
"""

from pathlib import Path
from sys import float_info
import pandas as pd
import numpy as np
import mne
import nibabel as nib
import statistics


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


def mni_to_mmp(mni_coord: np.ndarray, path_mmp: Path) -> list:
    """Get closest MMP parcel to MNI coordinate.

    Args:
        mni_coord (np.ndarray): MNI coordinates for each channel
        path_mmp (Path): path to .nii.gz file with MMP parcellation labels

    Returns:
        list: closest parcel for each channel
    """

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
    feature_name: str,
):
    """Compute averaged metric for each parcel.

    Args:
        df_patients (pd.DataFrame): dataframe with channels info and metric values
        output_grid (np.ndarray): matrix of non-zero voxels in the parcellation
        affine_transf (np.ndarray): affine transformation matrix
        d (float): distance in mm for Gaussian weighting (FWHM)
        feature_name (str): feature name in the dataframe.
    """

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

        # Create the weight matrix from input to output projection based on Gaussian weighting of Euclidean distance
        W_mat = np.zeros((input_grid.shape[0], output_grid.shape[0]))
        for ig in range(input_grid.shape[0]):
            W_mat[ig, :] = np.exp(
                -np.linalg.norm(output_grid - input_grid[ig, :], axis=1) ** 2
                / d_alpha**2
            )

        # Get total and max weights to drop bad coverage points
        W_max.append(np.max(W_mat, axis=0))
        feat_weighted.append(
            np.dot(df_patient[feature_name].values, W_mat)
            / np.where(W_mat.sum(axis=0) == 0, float_info.min, W_mat.sum(axis=0))
        )

    return feat_weighted, W_max


def compute_weighted_average(
    df_feature: pd.DataFrame,
    df_W: pd.DataFrame,
    w_thresh=0.5,
    axis=0,
    method="weighted",
):
    """Compute weighted average of feature values.

    Args:
        df_feature (pd.DataFrame): dataframe with feature value.
        df_W (pd.DataFrame): dataframe with weights.
        w_thresh (float, optional): threshold for weights. Defaults to 0.5.
        axis (int, optional): axis to average (patient or parcel). Defaults to 0.
        method (str, optional): method for averaging. Defaults to "weighted".

    Raises:
        ValueError: if method is not implemented.

    Returns:
        pd.DataFrame: dataframe with averaged feature values.
    """

    if method == "weighted":
        # method 1: weighted average of all parcels
        return (df_feature * df_W).sum(axis=axis) / np.where(
            df_W.sum(axis=axis) == 0,
            float_info.min,
            df_W.sum(axis=axis),
        )

    elif method == "thresh_weighted":
        # method 2: weighted average of suprathreshold parcels
        thresh_mat = df_W >= w_thresh
        return (df_feature * df_W)[thresh_mat].sum(axis=axis) / np.where(
            df_W[thresh_mat].sum(axis=axis) == 0,
            float_info.min,
            df_W[thresh_mat].sum(axis=axis),
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
    """Compute weights per parcel.

    Args:
        feat_weighted (list): feature weighted.
        W_max (list): max weight per patient.
        parcels (np.ndarray): array with parcels.
        parcels_names (np.ndarray): parcel names.
        pats (np.ndarray): patient codes.
        method (str, optional): method for averaging. Defaults to "weighted".

    Returns:
        dataframe with weights and feature averages.
    """

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
