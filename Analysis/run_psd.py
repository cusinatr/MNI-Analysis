"""
Script for computing timescales and logginf analysis parameters.
"""

from pathlib import Path
import yaml
from mnitimescales import PipePSD

###
# Analysis parameters
###

# Matlab file to use
mat_file = "MatlabFile.mat"  # "MatlabFile.mat", "NREM-sleep-20min.mat"

# Stages to analyze
stages = ["W", "N2", "N3", "R"]

# Output folder
out_dir = "psd_gamma_exp"

# Epochs duration
epo_dur = 1  # s
epo_overlap = 0.5  # s

# Frequencies for filtering
filt = False
filt_freqs = [0, 80]  # Hz

# ACF fit parameters
nlags = 100  # compute all lags
fit_mode = "fixed"  # fixed, knee
fit_range = [40, 80]  # Hz

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", mat_file)
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

pipe_psd = PipePSD(
    mat_path=mat_path,
    results_path=results_path,
    config_path=config_path,
    parc_path=parc_path,
    stages=stages
)
pipe_psd.run(
    epo_dur=epo_dur,
    epo_overlap=epo_overlap,
    filt=filt,
    filt_freqs=filt_freqs,
    fit_mode=fit_mode,
    fit_range=fit_range,
)
