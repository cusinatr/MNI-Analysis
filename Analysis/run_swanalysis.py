from pathlib import Path
from mnitimescales import PipeSW

###
# Analysis parameters
###

# Output folder
out_dir = "test_sw_pipe"

# Filter frequencies
sw_freqs = (0.5, 4)  # Hz
gamma_freqs = (30, 80)  # Hz

# Threshold for slow-waves duration
dur_threshold = (0.5, 2)  # s
dur_neg = (0.1, 2)  # s
dur_pos = (0.1, 2)  # s

# Biggest n% amplitudes
amp_percentile = 25

# Slow-waves epoch creation parameters
center_sws = "NegPeak"
t_epo_sws = 2  # s

###
# Paths
###

base_path = Path("F:\\MNIOpen")
mat_path = base_path.joinpath("Data", "Raw", "NREM-sleep-20min.mat")
results_path = base_path.joinpath("Results", out_dir)
parc_path = base_path.joinpath("Data", "Parcellation")

###
# Run analysis
###

results_path.mkdir(parents=True, exist_ok=True)

pipe_sw = PipeSW(
    mat_path=mat_path,
    results_path=results_path,
    parc_path=parc_path,
    stages=["N2", "N3"]
)
pipe_sw.run(
    sw_freqs=sw_freqs,
    gamma_freqs=gamma_freqs,
    dur_threshold=dur_threshold,
    dur_neg=dur_neg,
    dur_pos=dur_pos,
    amp_percentile=amp_percentile,
    center_sws=center_sws,
    t_epo_sws=t_epo_sws,
)
