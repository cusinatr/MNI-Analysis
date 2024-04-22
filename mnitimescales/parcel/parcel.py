import numpy as np
import pandas as pd
from .utils import (
    compute_parc_metric,
    compute_parc_metric_weight,
    compute_weighted_average,
)


# TODO: adjust function
# + think of doing a single class for different parcellations (MMP, MMP macro, MNI), i.e. could use 3 different functions and have some init tasks like loading the regions &
# + could in principle contain a method to also load & transform the MNI-Destrieux atlas


# Read data & region info for MMP
mmp_aggr = pd.read_csv(
    data_path.joinpath("mmp_macro.csv"),
    index_col=0,
    names=["parcel", "macro_region", "macro_labels"],
)

###
# Import MMP map in MNI coords
###

MMP_data = nib.load(data_path.joinpath("MMP_in_MNI_symmetrical_1.nii.gz"))
MMP_map = MMP_data.get_fdata()
MMP_map_nonzero = np.array(np.where(MMP_map > 0)).T
MMP_map_nonzero_flat = MMP_map[np.where(MMP_map > 0)]

# Retrieve annotations files
subjects_dir = str(mne.datasets.sample.data_path()) + "/subjects"
parc_map_name = "HCPMMP1"
annot_file = subjects_dir + "/fsaverage/label/lh." + parc_map_name + ".annot"
parc_labels, ctab, parc_names = nib.freesurfer.read_annot(annot_file)
region_names = [n[2:-4].decode("utf-8") for n in parc_names[1:]]

###
# Compute all aggregated metrics
###


def parcel_mmp(
    df_metric: pd.DataFrame,
    metric_name: str,
    d=4,
):

    # Compute values for each subject and voxel of the parcellation
    metric_weighted, W_max = compute_parc_metric(
        df_metric, MMP_map_nonzero, MMP_data.affine, d, feature=metric_name
    )
    # Collapse across subjects and voxels
    (
        df_W_max_avg,
        df_metric_weighted_avg,
        df_metric_parc,
    ) = compute_parc_metric_weight(
        metric_weighted,
        W_max,
        MMP_map_nonzero_flat,
        region_names,
        df_metric["pat"],
    )

    # Compute also metrics for "macro" regions
    df_metric_macro_mmp = pd.DataFrame(columns=np.unique(mmp_aggr["macro_labels"]))
    for i_r, r in mmp_aggr.groupby("macro_labels"):
        df_metric_macro_mmp[i_r] = compute_weighted_average(
            df_metric_weighted_avg[r["parcel"]],
            df_W_max_avg[r["parcel"]],
            axis=1,
            method="thresh_weighted",
        )
    df_metric_macro_mmp = pd.DataFrame(columns=["mean", "sem"])
    df_metric_macro_mmp["mean"] = df_metric_macro_mmp.mean(axis=0)
    df_metric_macro_mmp["sem"] = df_metric_macro_mmp.sem(axis=0)

    return df_metric_parc, df_metric_macro_mmp


def parcel_mni(df_metric: pd.DataFrame):

    # Read data & region info from MNI
    df_regions_mni = pd.read_csv(data_path.joinpath("RegionInformation.csv"))
    df_regions_mni["Region name"] = df_regions_mni["Region name"].apply(
        lambda x: x.strip("'")
    )

    # MNI macro
    df_metric_macro_mni = []
    for lobe in df_regions_mni["Lobe"].unique():
        df_lobe = df_regions_mni[df_regions_mni["Lobe"] == lobe]
        df_metric_lobe = df_metric[
            df_metric["region"].isin(df_lobe["Region name"])
        ]
        df_metric_lobe["lobe"] = lobe
        df_metric_macro_mni.append(df_metric_lobe)
    df_metric_macro_mni = pd.concat(df_metric_macro_mni, ignore_index=True)

# # Save results
# df_W_max_avg_stage.to_csv(res_path.joinpath(f"W_max_{stage}.csv"))
# df_tau_weighted_avg_stage.to_csv(
#     res_path.joinpath(f"{save_name}_weighted_{stage}.csv")
# )
# df_tau_parc_stage.to_csv(res_path.joinpath(f"{save_name}_parc_{stage}.csv"))
# df_tau_macro_stage_mmp.to_csv(
#     res_path.joinpath(f"{save_name}_macro_mmp_{stage}.csv")
# )
# df_tau_macro_stage_mni.to_csv(
#     res_path.joinpath(f"{save_name}_macro_mni_{stage}.csv")
# )
