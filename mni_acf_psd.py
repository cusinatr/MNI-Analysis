from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
from ieeganalysis import PSD, SpectralParam, ACF, TAU
from mni_utils import convert_knee_tau

###
# Analysis parameters
###

# Epochs parameters
epo_dur = 1
epo_overlap = 0.5
# ACF / Tau parameters
compute_acf = False
nlags = 100
tau_mode = "fit"  # fit, interp
# PSD / SpectralParam parameters
compute_psd = True
fit_mode = "fixed"  # knee, fixed
frange_fit = [30, 45]

# Output folder
out_dir = "Results_psd_exp_30-45"    # needs to match that in .yml file

###
# Paths
###

base_path = Path("D:\\iEEG_neural_dynamics\\MNIOpen")
res_path = base_path.joinpath(out_dir)
res_path.mkdir(parents=True, exist_ok=True)
config_path = Path(__file__).parent.joinpath("config_mni.yml")

###
# Import data
###

data_path = base_path.joinpath("MatlabFile.mat")
data = loadmat(data_path)
data_W = data["Data_W"].T
data_N2 = data["Data_N2"].T
data_N3 = data["Data_N3"].T
data_R = data["Data_R"].T

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
# Loop through patients to compute measures
###

pats = df_info["pat"].unique().tolist()
df_tau_pats = []
df_knee_pats = []

for i_p, pat in enumerate(pats):
    print("Patient: ", pat)

    pat_code = str(pat).zfill(3)
    df_info_pat = df_info[df_info["pat"] == pat]

    # Dataframe with results
    df_tau_pat = []
    df_knee_pat = []

    # Create folders & files
    pat_path = res_path.joinpath(pat_code)
    pat_path.mkdir(parents=True, exist_ok=True)
    proc_path = pat_path.joinpath("ProcData")
    proc_path.mkdir(parents=True, exist_ok=True)

    # Create info file
    df_meta_pat = pd.DataFrame(index=df_info_pat["chan"], columns=["lead", "bad"])
    df_meta_pat["lead"] = "depth"
    df_meta_pat["bad"] = False
    df_meta_pat.to_csv(pat_path.joinpath(pat_code + "_meta.csv"))

    # Prepare data analyses
    patPSD = PSD(config_path, pat_code)
    patSP = SpectralParam(config_path, pat_code)
    patACF = ACF(config_path, pat_code)
    patTAU = TAU(config_path, pat_code)

    chans_pat = df_info_pat["chan"].to_list()

    # Compute psd and knee
    for stage, data_stage in zip(
        ["W", "N2", "N3", "R"], [data_W, data_N2, data_N3, data_R]
    ):
        print(stage)
        data_stage_pat = data_stage[df_info_pat.index]

        # Dataframe for stage
        df_tau_pat_stage = pd.DataFrame(
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
                "tau",
            ]
        )
        df_tau_pat_stage["pat"] = [pat_code] * len(df_info_pat)
        df_tau_pat_stage["age"] = [age[i_p]] * len(df_info_pat)
        df_tau_pat_stage["gender"] = [gender[i_p][0]] * len(df_info_pat)
        df_tau_pat_stage["chan"] = df_info_pat["chan"].to_list()
        df_tau_pat_stage["type"] = df_info_pat["type"].to_list()
        df_tau_pat_stage["stage"] = [stage] * len(df_info_pat)
        df_tau_pat_stage["region"] = df_info_pat["region"].to_list()
        df_tau_pat_stage["mni_x"] = df_info_pat["mni_x"].to_list()
        df_tau_pat_stage["mni_y"] = df_info_pat["mni_y"].to_list()
        df_tau_pat_stage["mni_z"] = df_info_pat["mni_z"].to_list()
        df_knee_pat_stage = pd.DataFrame(
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
                "tau",
                "exp",
                "r2",
            ]
        )
        df_knee_pat_stage["pat"] = [pat_code] * len(df_info_pat)
        df_knee_pat_stage["age"] = [age[i_p]] * len(df_info_pat)
        df_knee_pat_stage["gender"] = [gender[i_p][0]] * len(df_info_pat)
        df_knee_pat_stage["chan"] = df_info_pat["chan"].to_list()
        df_knee_pat_stage["type"] = df_info_pat["type"].to_list()
        df_knee_pat_stage["stage"] = [stage] * len(df_info_pat)
        df_knee_pat_stage["region"] = df_info_pat["region"].to_list()
        df_knee_pat_stage["mni_x"] = df_info_pat["mni_x"].to_list()
        df_knee_pat_stage["mni_y"] = df_info_pat["mni_y"].to_list()
        df_knee_pat_stage["mni_z"] = df_info_pat["mni_z"].to_list()

        # Detect nan channels
        chans_nan = np.array(chans_pat)[
            np.where(np.all(np.isnan(data_stage_pat), axis=1))[0]
        ]
        chans_good = [ch for ch in chans_pat if ch not in chans_nan]

        if chans_good:
            idx_good = [i for i, ch in enumerate(chans_pat) if ch in chans_good]
            idx_nan = [i for i, ch in enumerate(chans_pat) if ch in chans_nan]
            info_stage_pat = mne.create_info(chans_good, sfreq, ch_types="seeg")
            raw_stage_pat = mne.io.RawArray(data_stage_pat[idx_good], info_stage_pat)

            # Detect flat time segments
            idx_flat = np.where(np.all(raw_stage_pat._data == 0, axis=0))[0]
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
                raw_stage_pat.set_annotations(flat_annot)

            # Epochs
            epo_stage_pat = mne.make_fixed_length_epochs(
                raw_stage_pat, duration=epo_dur, overlap=epo_overlap, preload=True
            )

            # ACF
            if compute_acf:
                patACF.prepare_data(epochs=epo_stage_pat, stage=stage)
                acf_stage_pat = patACF.compute_acf(nlags=nlags, label="")
                # TAU
                patTAU.prepare_data(acf_stage_pat, stage=stage, label=stage)
                tau_stage_pat = patTAU.compute_timescales(
                    mode=tau_mode, fit_func="exp_oscill", fit_range=[0.001, 0.3]
                )
                patTAU.plot_timescales()
                # Save data
                df_tau_pat_stage.loc[idx_good, "tau"] = [
                    v for k, v in tau_stage_pat.items() if k in chans_good
                ]
                df_tau_pat_stage.loc[idx_nan, "tau"] = np.nan
            # PSD
            if compute_psd:
                patPSD.prepare_data(epochs=epo_stage_pat, stage=stage)
                psd_stage_pat = patPSD.compute_psd(f_band=[1, 80], label="")
                # Spectral Param
                patSP.prepare_data(psd_stage_pat, stage=stage, label=stage)
                fg_stage_pat = patSP.parametrize_psd(
                    frange=frange_fit, aperiodic_mode=fit_mode, save_full=False
                )
                patSP.plot_parametrization(plot_range=[1, 80])
                # Save data
                if fit_mode == "knee":
                    knee_stage_pat = convert_knee_tau(
                        {
                            "knee": fg_stage_pat.get_params("aperiodic_params", "knee"),
                            "exp": fg_stage_pat.get_params("aperiodic_params", "exponent"),
                        }
                    )
                else:
                    knee_stage_pat = np.array([np.nan] * len(fg_stage_pat))
                exp_stage_pat = fg_stage_pat.get_params("aperiodic_params", "exponent")
                r2_stage_pat = fg_stage_pat.get_params("r_squared")
            
                df_knee_pat_stage.loc[idx_good, "tau"] = knee_stage_pat
                df_knee_pat_stage.loc[idx_nan, "tau"] = np.nan
                df_knee_pat_stage.loc[idx_good, "exp"] = exp_stage_pat
                df_knee_pat_stage.loc[idx_nan, "exp"] = np.nan
                df_knee_pat_stage.loc[idx_good, "r2"] = r2_stage_pat
                df_knee_pat_stage.loc[idx_nan, "r2"] = np.nan

        else:
            df_tau_pat_stage["tau"] = np.nan
            df_knee_pat_stage["tau"] = np.nan

        df_tau_pat.append(df_tau_pat_stage)
        df_knee_pat.append(df_knee_pat_stage)

    # Save results
    df_tau_pat = pd.concat(df_tau_pat, ignore_index=True)
    df_tau_pat.to_csv(patTAU.acf_dir.joinpath(pat_code + "_tau.csv"))
    df_knee_pat = pd.concat(df_knee_pat, ignore_index=True)
    df_knee_pat.to_csv(patSP.psd_dir.joinpath(pat_code + "_knee.csv"))

    df_tau_pats.append(df_tau_pat)
    df_knee_pats.append(df_knee_pat)

# Save all results
df_tau_pats = pd.concat(df_tau_pats, ignore_index=True)
df_tau_pats.to_csv(res_path.joinpath("all_tau.csv"))

df_knee_pats = pd.concat(df_knee_pats, ignore_index=True)
df_knee_pats.to_csv(res_path.joinpath("all_knee.csv"))