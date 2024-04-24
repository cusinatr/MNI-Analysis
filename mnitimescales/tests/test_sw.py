from pathlib import Path
from mnitimescales import Load, ComputeSW

# Load info dataframe
load = Load(mat_path="F:\\MNIOpen\\Data\\Raw\\NREM-sleep-20min.mat")
df_info = load.get_info()

# Load data
raw_stage = load.load_raw_stage("N2")

# Compute SWs
results_path = Path("F:\\MNIOpen\\Results\\test_sw_log")
results_path.mkdir(exist_ok=True)
compute_sw = ComputeSW(
    df_info,
    raw_stage,
    "N2",
    results_path,
)
compute_sw.detect_sw()

