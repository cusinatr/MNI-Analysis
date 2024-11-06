from pathlib import Path
import pandas as pd
from mnitimescales import Load, ComputeSW, Parcel

# Load info dataframe
load = Load(mat_path="F:\\MNIOpen\\Data\\Raw\\NREM-sleep-20min.mat")
df_info = load.get_info()
df_info = df_info[df_info["pat"] == "098"]

# Load data
raw_stage = load.load_raw_stage("N3")

# Compute SWs
results_path = Path("F:\\MNIOpen\\Results\\test_sw_pipe")
# df_density = pd.read_csv(results_path.joinpath("density_N2.csv"), index_col=0)
# parc_path = Path("F:\\MNIOpen\\Data\\Parcellation")

compute_sw = ComputeSW(
    df_info,
    raw_stage,
    "N3",
    results_path,
)
df_density = compute_sw.detect_sw()

# for cond in ["total", "local", "global"]:
#     parc = Parcel(parc_path=parc_path)
#     df_density_mmp, df_density_mmp_macro = parc.parcel_mmp(df_density, cond)
#     df_density_mni = parc.parcel_mni(df_density)


