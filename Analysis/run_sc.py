"""
Script for computing spatial correlation parameters and logging analysis.
"""

from pathlib import Path
import pandas as pd
from mnitimescales import PipeSC

###
# Analysis parameters
###

# Matlab file to use
mat_file = "MatlabFile.mat"  # "MatlabFile.mat", "NREM-sleep-20min.mat"

# Stages to analyze
stages = ["W", "N2", "N3", "R"]

# Output folder
out_dir = "sc_analysis_broadband"

# Epochs duration
epo_dur = 1  # s
epo_overlap = 0.5  # s

# Frequencies for filtering
filt = False
filt_freqs = [0, 80]  # Hz

# SC parameters
use_bispectrum = False
fit_modes = ["fit", "fit", "median", "median"]
col_names = ["corr", "lag", "corr", "lag"]

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", mat_file)
regions_path = base_path.joinpath("Data", "Parcellation", "RegionInformation.csv")
results_path = base_path.joinpath("Results", out_dir)


###
# Run analysis
###

results_path.mkdir(parents=True, exist_ok=True)
df_regions = pd.read_csv(regions_path)
df_regions["Region name"] = df_regions["Region name"].apply(lambda x: x.strip("'"))

pipe_sc = PipeSC(
    mat_path=mat_path, results_path=results_path, df_regions=df_regions, stages=stages
)
pipe_sc.run_compute(
    epo_dur=epo_dur,
    epo_overlap=epo_overlap,
    filt=filt,
    filt_freqs=filt_freqs,
    use_bispectrum=use_bispectrum,
)
pipe_sc.run_fit(fit_modes=fit_modes, col_names=col_names)
