from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from .utils import (
    compute_parc_metric,
    compute_parc_metric_weight,
    compute_weighted_average,
)


# TODO: method to also load & transform the MNI-Destrieux atlas


class Parcel:

    def __init__(self, parc_path: Path):

        # Import atlases for parcellation
        # HCP-MMP atlas
        subjects_dir = str(mne.datasets.sample.data_path()) + "/subjects"
        parc_map_name = "HCPMMP1"
        annot_file = subjects_dir + "/fsaverage/label/lh." + parc_map_name + ".annot"
        _, _, parc_names = nib.freesurfer.read_annot(annot_file)
        self.mmp_region_names = [n[2:-4].decode("utf-8") for n in parc_names[1:]]

        # MMP map in MNI coords
        self.MMP_data = nib.load(parc_path.joinpath("MMP_in_MNI_symmetrical_1.nii.gz"))
        MMP_map = self.MMP_data.get_fdata()
        self.MMP_map_nonzero = np.array(np.where(MMP_map > 0)).T
        self.MMP_map_nonzero_flat = MMP_map[np.where(MMP_map > 0)]

        # Macro regions MMP
        self.mmp_aggr = pd.read_csv(
            parc_path.joinpath("mmp_macro.csv"),
            index_col=0,
            names=["parcel", "macro_region", "macro_labels"],
        )

        # Read data & region info from MNI
        self.df_regions_mni = pd.read_csv(parc_path.joinpath("RegionInformation.csv"))
        self.df_regions_mni["Region name"] = self.df_regions_mni["Region name"].apply(
            lambda x: x.strip("'")
        )

    def parcel_mmp(self, df_metric: pd.DataFrame, metric_name: str, d=4):

        # Compute values for each subject and voxel of the parcellation
        metric_weighted, W_max = compute_parc_metric(
            df_metric,
            self.MMP_map_nonzero,
            self.MMP_data.affine,
            d,
            feature=metric_name,
        )
        # Collapse across subjects and voxels
        (
            df_W_max_avg,
            df_metric_weighted_avg,
            df_metric_parc,
        ) = compute_parc_metric_weight(
            metric_weighted,
            W_max,
            self.MMP_map_nonzero_flat,
            self.mmp_region_names,
            df_metric["pat"],
        )

        # Compute also metrics for "macro" regions
        df_metric_macro_mmp = pd.DataFrame(
            columns=np.unique(self.mmp_aggr["macro_labels"])
        )
        for i_r, r in self.mmp_aggr.groupby("macro_labels"):
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

    def parcel_mni(self, df_metric: pd.DataFrame):

        # MNI macro
        df_metric_macro_mni = []
        for lobe in self.df_regions_mni["Lobe"].unique():
            df_lobe = self.df_regions_mni[self.df_regions_mni["Lobe"] == lobe]
            df_metric_lobe = df_metric[df_metric["region"].isin(df_lobe["Region name"])]
            df_metric_lobe["lobe"] = lobe
            df_metric_macro_mni.append(df_metric_lobe)
        df_metric_macro_mni = pd.concat(df_metric_macro_mni, ignore_index=True)

        return df_metric_macro_mni
