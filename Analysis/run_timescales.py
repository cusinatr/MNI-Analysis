from datetime import datetime
from pathlib import Path
import yaml
from mnitimescales import PipeTC

###
# Analysis parameters
###

# Epochs duration
epo_dur = 1  # s
epo_overlap = 0.5  # s

# Frequencies for filtering
filt = False
filt_freqs = [40, 80]  # Hz

# ACF fit parameters
nlags = 100  # compute all lags
tau_mode = "fit"  # fit, interp
fit_func = "exp"  # exp, exp_oscill, exp_double
fit_range = [0.001, 0.5]

###
# Paths
###

# Output folder
out_dir = "test"

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", "MatlabFile.mat")
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

pipe_timescales = PipeTC(
    mat_path=mat_path,
    results_path=results_path,
    config_path=config_path,
    parc_path=parc_path,
)
pipe_timescales.run(
    epo_dur=epo_dur,
    epo_overlap=epo_overlap,
    filt=filt,
    filt_freqs=filt_freqs,
    nlags=nlags,
    tau_mode=tau_mode,
    fit_func=fit_func,
    fit_range=fit_range,
)
