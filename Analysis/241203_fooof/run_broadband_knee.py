"""
Script for computing timescales from PSD and logging analysis parameters.
"""

from pathlib import Path
import yaml
from mnitimescales import PipePSD

###
# Analysis parameters
###

# Matlab file to use
mat_file = "MatlabFile.mat"

# Stages to analyze
stages = ["W", "N3", "R"]

# Output folder
out_dir = "timescales_broadband_knee_1_80"

# Epochs duration
epo_dur = 1  # s
epo_overlap = 0.5  # s

# Frequencies for filtering
filt = False
filt_freqs = []  # Hz

# PSD fit parameters
fit_mode = "knee"
fit_range = [1, 80]  # Hz

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", mat_file)
results_path = base_path.joinpath("Results", out_dir)
config_path = Path(__file__).parent.joinpath("configs.yml")
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
    plot=True
)
