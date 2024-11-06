"""
Compute timescales during and before/after a slow-wave.
"""

from pathlib import Path
import yaml
from mnitimescales import PipeTCSW

###
# Analysis parameters
###

stages = ["N3"]

# Input folders
sw_dir = "sw_analysis"

# Output folder
out_dir = "sw_timescales_gamma"

# Epochs duration
epo_dur = 1  # s
epo_overlap = 0.5  # s
epo_sws = 2.5  # s

# Frequencies for filtering for timescales
filt = True
filt_freqs = [40, 80]  # Hz

# ACF fit parameters
nlags = 100  # compute all lags
tau_mode = "fit"  # fit, interp
fit_func = "exp"  # exp, exp_oscill, exp_double
fit_range = [0.015, 0.3]  # [0.001, 0.5]  # [0.015, 0.3]

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", "NREM-sleep-20min.mat")
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

pipe_tc_sw = PipeTCSW(
    mat_path=mat_path,
    sw_path=sw_path,
    results_path=results_path,
    config_path=config_path,
    parc_path=parc_path,
    stages=stages,
)
pipe_tc_sw.run(
    epo_sws=epo_sws,
    epo_dur=epo_dur,
    epo_overlap=epo_overlap,
    filt=filt,
    filt_freqs=filt_freqs,
    nlags=nlags,
    tau_mode=tau_mode,
    fit_func=fit_func,
    fit_range=fit_range,
)
