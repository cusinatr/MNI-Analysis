"""
Script for computing parcellation averages of timescales.
For now, only atlas implemented is the MMP.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import mne

import mni_utils as uti


###
# Paths and names
###

data_path = Path("F:\\iEEG_neural_dynamics\\MNIOpen")
res_dir = "Results_acf_fit_exp"
data_name = "all_tau.csv"
metric_name = "tau"
save_name = "tau"

###
# Import timescales dataframe
###

res_path = data_path.joinpath(res_dir)
df_tau = pd.read_csv(res_path.joinpath(data_name), index_col=0)

###
# Filter dataframe
###

if metric_name != "tau":
    df_tau.drop(columns=["tau"], inplace=True)
df_tau = df_tau.dropna()
if "r2" in df_tau.columns:
    df_tau = df_tau[df_tau["r2"] > 0.8]

###
# Import regions dataframe
###

# Read data & region info from MNI
df_regions_mni = pd.read_csv(data_path.joinpath("RegionInformation.csv"))
df_regions_mni["Region name"] = df_regions_mni["Region name"].apply(
    lambda x: x.strip("'")
)

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
annot_file = (
    subjects_dir + "/fsaverage/label/lh." + parc_map_name + ".annot"
)
parc_labels, ctab, parc_names = nib.freesurfer.read_annot(annot_file)
region_names = [n[2:-4].decode("utf-8") for n in parc_names[1:]]

###
# Compute all aggregated metrics
###

# Loop over stages
for stage in df_tau["stage"].unique():

    # Compute values for each subject and voxel of the parcellation
    df_tau_stage = df_tau[df_tau["stage"] == stage]
    tau_weighted_stage, W_max_stage = uti.compute_parc_metric(
        df_tau_stage, MMP_map_nonzero, MMP_data.affine, d=4, feature=metric_name
    )
    # Collapse across subjects and voxels
    (
        df_W_max_avg_stage,
        df_tau_weighted_avg_stage,
        df_tau_parc_stage,
    ) = uti.compute_parc_metric_weight(
        tau_weighted_stage,
        W_max_stage,
        MMP_map_nonzero_flat,
        region_names,
        df_tau_stage["pat"],
    )

    # Compute also metrics for "macro" regions
    df_tau_macro_stage = pd.DataFrame(columns=np.unique(mmp_aggr["macro_labels"]))
    for i_r, r in mmp_aggr.groupby("macro_labels"):
        df_tau_macro_stage[i_r] = uti.compute_weighted_average(
            df_tau_weighted_avg_stage[r["parcel"]],
            df_W_max_avg_stage[r["parcel"]],
            axis=1,
            method="thresh_weighted",
        )
    df_tau_macro_stage_mmp = pd.DataFrame(columns=["mean", "sem"])
    df_tau_macro_stage_mmp["mean"] = df_tau_macro_stage.mean(axis=0)
    df_tau_macro_stage_mmp["sem"] = df_tau_macro_stage.sem(axis=0)

    # MNI macro
    df_tau_macro_stage_mni = []
    for lobe in df_regions_mni["Lobe"].unique():
        df_lobe = df_regions_mni[df_regions_mni["Lobe"] == lobe]
        df_tau_stage_lobe = df_tau_stage[
            df_tau_stage["region"].isin(df_lobe["Region name"])
        ]
        df_tau_stage_lobe["lobe"] = lobe
        df_tau_macro_stage_mni.append(df_tau_stage_lobe)
    df_tau_macro_stage_mni = pd.concat(df_tau_macro_stage_mni, ignore_index=True)

    # Save results
    df_W_max_avg_stage.to_csv(res_path.joinpath(f"W_max_{stage}.csv"))
    df_tau_weighted_avg_stage.to_csv(
        res_path.joinpath(f"{save_name}_weighted_{stage}.csv")
    )
    df_tau_parc_stage.to_csv(res_path.joinpath(f"{save_name}_parc_{stage}.csv"))
    df_tau_macro_stage_mmp.to_csv(
        res_path.joinpath(f"{save_name}_macro_mmp_{stage}.csv")
    )
    df_tau_macro_stage_mni.to_csv(
        res_path.joinpath(f"{save_name}_macro_mni_{stage}.csv")
    )
