"""
Compute timescales during and before/after a slow-wave.
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import mne
from mnitimescales import Load
from ieeganalysis import ACF, TAU
import mnitimescales.sw.utils as utils_sw
import mnitimescales.fit.utils as utils_acf

###
# Analysis parameters
###

stage = "N3"

# Input folders
sw_dir = "sw_analysis"

# Output folder
out_dir = "timescales_sw_test_short_windows"

# Epochs duration
epo_dur = 0.5  # s
epo_overlap = 0.25  # s
epo_sws = 2  # s

# Frequencies for filtering for timescales
filt = False
filt_freqs = [40, 80]  # Hz

# ACF fit parameters
nlags = 50  # compute all lags
tau_mode = "fit"  # fit, interp
fit_func = "exp"  # exp, exp_oscill, exp_double
fit_range = [0.001, 0.25]  # [0.015, 0.3]

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath(
    "Data", "Raw", "NREM-sleep-20min.mat"
)  # "MatlabFile.mat")
sw_path = base_path.joinpath("Results", sw_dir, "Pats")
results_path = base_path.joinpath("Results", out_dir)
config_path = Path(__file__).parent.joinpath("config_mni.yml")
parc_path = base_path.joinpath("Data", "Parcellation")


###
# Run analysis
###

with open(config_path) as f:
    configs = yaml.safe_load(f)
configs["base_path"] = results_path.joinpath("Pats").as_posix()
with open(config_path, "w") as f:
    yaml.dump(configs, f)

results_path.mkdir(parents=True, exist_ok=True)

# Load info dataframe & stage data
load = Load(mat_path=mat_path)
df_info = load.get_info()
raw_stage = load.load_raw_stage(stage)

# Dataframe for all results
df_timescales_sw = []
df_acf_sw = []

for pat in raw_stage.keys():

    df_timescales_sw_pat = []
    df_acf_sw_pat = []

    raw_pat = raw_stage[pat]
    df_events = pd.read_csv(
        sw_path.joinpath(pat, f"sw_events_{stage}.csv"), index_col=0
    )
    epochs_pat = utils_sw._epoch_sws(
        df_events,
        raw_pat.get_data(),
        raw_pat.ch_names,
        raw_pat.info["sfreq"],
        center="NegPeak",
        t_around=epo_sws,
    )

    # Create folders & files
    utils_acf.create_pat_folder(pat, results_path, raw_pat.ch_names)

    # Prepare data analyses
    patACF = ACF(config_path, pat)
    patTAU = TAU(config_path, pat)

    for chan in epochs_pat.keys():
        epo_chan = epochs_pat[chan]
        t_epo = np.linspace(-epo_sws, epo_sws, epo_chan.shape[1])
        info_chan = mne.create_info([chan], raw_pat.info["sfreq"], ch_types="seeg")
        for t_start in np.arange(-epo_sws, epo_sws, epo_overlap):
            t_end = t_start + epo_dur
            if t_end >= epo_sws:
                continue
            epo_chan_sw = epo_chan[:, (t_epo > t_start) & (t_epo <= t_end)]
            epo_chan_sw = mne.EpochsArray(
                np.expand_dims(epo_chan_sw, axis=1),
                info_chan,
                tmin=t_start,
                verbose=False,
            )
            # Compute ACF
            patACF.prepare_data(epochs=epo_chan_sw)  # , stage=self.stage)
            acf_pat = patACF.compute_acf(
                nlags=nlags, label=f"{chan}_{(t_start + t_end) / 2}"
            )
            # Compute Timescale
            patTAU.prepare_data(acf_pat, label=f"{chan}_{(t_start + t_end) / 2}")
            tau_pat = patTAU.compute_timescales(
                mode=tau_mode, fit_func=fit_func, fit_range=fit_range
            )

            df_timescales_sw_pat.append(
                pd.DataFrame(
                    {
                        "pat": pat,
                        "chan": chan,
                        "time": (t_start + t_end) / 2,
                        "tau": tau_pat[chan],
                    },
                    index=[0],
                )
            )
            df_acf_sw_pat.append(
                pd.DataFrame(
                    {
                        "pat": pat,
                        "chan": chan,
                        "time": (t_start + t_end) / 2,
                    } | {lag: val for lag, val in zip(patTAU.acf_data["lags"], patTAU.acf_data["acf_avg"][0])},
                    index=[0],
                )
            )
    df_timescales_sw_pat = pd.concat(df_timescales_sw_pat, ignore_index=True)
    df_timescales_sw.append(df_timescales_sw_pat)
    df_acf_sw_pat = pd.concat(df_acf_sw_pat, ignore_index=True)
    df_acf_sw.append(df_acf_sw_pat)

df_timescales_sw = pd.concat(df_timescales_sw, ignore_index=True)
df_acf_sw = pd.concat(df_acf_sw, ignore_index=True)

# Save results
df_timescales_sw.to_csv(results_path.joinpath(f"timescales_sw_{stage}.csv"))
df_acf_sw.to_csv(results_path.joinpath(f"acf_sw_{stage}.csv"))
