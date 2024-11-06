import pandas as pd
import numpy as np

for stage in ["W", "N3", "R"]:
    df_old = pd.read_csv(f"F:\\MNIOpen\\old\\Results_acf_fit_exp\\tau_parc_{stage}.csv", index_col=0)
    df_new = pd.read_csv(f"F:\\MNIOpen\\Results\\timescales_broadband\\tau_{stage}_mmp.csv", index_col=0)

    print(stage, np.isclose(df_old, df_new).all())