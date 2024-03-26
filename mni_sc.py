"""
Compute spatial correlations between electrodes and fit exponential decay.
"""

from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
import mni_utils as uti


###
# Paths and parameters
###

base_path = Path("F:\\iEEG_neural_dynamics\\MNIOpen")
mmp_name = "MMP_in_MNI_symmetrical_1.nii.gz"
out_dir = "Results_SC_bispectrum"
dist_bins = np.arange(0, 101, 10)  # bins to use as distances
freq_band = False
band_freqs = [40, 80]
recompute = True
fit_bins = False
use_bispectrum = True

###
# Create folder for storing results
###

res_path = base_path.joinpath(out_dir)
res_path.mkdir(parents=True, exist_ok=True)
mmp_path = base_path.joinpath(mmp_name)

###
# Import data
###

data_path = base_path.joinpath("MatlabFile.mat")
data = loadmat(data_path)

###
# Convert data to Dataframe
###

ch_names = data["ChannelName"].squeeze()
ch_types = data["ChannelType"].squeeze()
ch_regs = data["ChannelRegion"].squeeze()
pat_ids = data["Patient"].squeeze()
gender = data["Gender"].squeeze()
age = data["AgeAtTimeOfStudy"].squeeze()
ch_pos = data["ChannelPosition"].squeeze()
regions_map = {i + 1: r[0][0] for i, r in enumerate(data["RegionName"])}
sfreq = data["SamplingFrequency"][0][0]
df_info = pd.DataFrame(
    {
        "pat": pat_ids,
        "chan": [ch[0] for ch in ch_names],
        "type": [t[0] for t in ch_types],
        "region": ch_regs,
        "mni_x": ch_pos[:, 0],
        "mni_y": ch_pos[:, 1],
        "mni_z": ch_pos[:, 2],
    }
)

df_info["region"] = df_info["region"].apply(lambda x: regions_map[x])

###
# Read data & region info from MNI
###

df_regions_mni = pd.read_csv(base_path.joinpath("RegionInformation.csv"))
df_regions_mni["Region name"] = df_regions_mni["Region name"].apply(
    lambda x: x.strip("'")
)

###
# Compute cross-correlations
###

df_sc_stages = {}
df_sc_bin_stages = {}

# Loop through stages to compute cross-correlations
for stage in ["W", "N3", "R"]:
    print(stage)

    # Select stage data
    data_stage = data["Data_" + stage].T

    # Compute spatial correlation dataframe
    if recompute:
        df_sc_stages[stage] = uti.compute_SC(
            data_stage,
            df_info,
            df_regions_mni,
            sfreq,
            mmp_path,
            freq_band=freq_band,
            band_freqs=band_freqs,
            use_bispectrum=use_bispectrum,
        )
        # Save
        df_sc_stages[stage].to_csv(res_path.joinpath(f"SC_{stage}.csv"))
    else:
        df_sc_stages[stage] = pd.read_csv(
            res_path.joinpath(f"SC_{stage}.csv"), index_col=0
        )

    # Compute binned spatial correlations
    if recompute:
        df_sc_bin_stages[stage] = uti.compute_sc_bin(
            df_sc_stages[stage], bins=dist_bins
        )
        df_sc_bin_stages[stage].to_csv(res_path.joinpath(f"SC_{stage}_bins.csv"))
    else:
        df_sc_bin_stages[stage] = pd.read_csv(
            res_path.joinpath(f"SC_{stage}_bins.csv"), index_col=0
        )

    ###
    # Fit exponential decay to spatial correlations
    ###

    if use_bispectrum:
        upper_bounds = (100, np.inf, np.inf)
    else:
        upper_bounds = (100, 1, 1)

    # First, fit to "global" profile
    if fit_bins:
        popt, pcov = uti.fit_sc_bins(df_sc_bin_stages[stage], upper_bounds=upper_bounds)
    else:
        popt, pcov = uti.fit_sc(df_sc_stages[stage], upper_bounds=upper_bounds)
    df_params = pd.DataFrame(popt.reshape(1, -1), columns=["k", "a", "b"])
    file_name = f"SC_{stage}_fit"
    file_name += "_bins" if fit_bins else ""
    df_params.to_csv(res_path.joinpath(file_name + ".csv"))

    # Then, one fit per MNI region
    # First on max CC value
    df_params = []
    for reg in df_sc_stages[stage]["region_1"].unique():
        df_sc_reg = df_sc_stages[stage][
            (df_sc_stages[stage]["region_1"] == reg)
            | (df_sc_stages[stage]["region_2"] == reg)
        ]
        if fit_bins:
            df_sc_bin_reg = uti.compute_sc_bin(df_sc_reg, bins=dist_bins)
            popt, pcov = uti.fit_sc_bins(df_sc_bin_reg, upper_bounds=upper_bounds)
        else:
            popt, pcov = uti.fit_sc(df_sc_reg, upper_bounds=upper_bounds)
        df_params.append(
            pd.DataFrame(popt.reshape(1, -1), columns=["k", "a", "b"], index=[reg])
        )
    df_params = pd.concat(df_params)
    file_name = f"SC_{stage}_fit_mni_regs"
    file_name += "_bins" if fit_bins else ""
    df_params.to_csv(res_path.joinpath(file_name + ".csv"))
