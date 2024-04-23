from mnitimescales.parcel import Parcel
from pathlib import Path
import pandas as pd

parc = Parcel(parc_path=Path("F:\\MNIOpen\\Data\\Parcellation"))

df_tau = pd.read_csv(Path("F:\\MNIOpen\\old\\Results_acf_fit_exp\\all_tau.csv"), index_col=0)
df_tau = df_tau[df_tau["stage"] == "W"]

df_tau_mmp, df_tau_mmp_macro = parc.parcel_mmp(df_tau, "tau")
df_tau_mni = parc.parcel_mni(df_tau)

print()