from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ieegtausleep.plots import plot_conds_regs


data_path = Path("D:\\iEEG_neural_dynamics\\MNIOpen\\Results")
data_name = "all_tau.csv"
fig_name = "Tau_comp"
ylim = [0, 300]
yscale = "linear"

# Read data & region info
df_tau = pd.read_csv(data_path.joinpath(data_name), index_col=0)
df_regions = pd.read_csv(data_path.parent.joinpath("RegionInformation.csv"))
df_regions["Region name"] = df_regions["Region name"].apply(lambda x: x.strip("'"))

# Filter nans & high values
df_tau = df_tau.dropna()
df_tau = df_tau[df_tau["tau"] < 500]


# Change names
df_tau.rename(columns={"stage": "cond", "tau": "Timescale [ms]"}, inplace=True)

# # Log
# df_tau["Timescale [ms]"] = df_tau["Timescale [ms]"].apply(lambda x: np.log10(x))

# Plot one figure per lobe
for lobe in df_regions["Lobe"].unique():
    df_lobe = df_regions[df_regions["Lobe"] == lobe]
    df_tau_lobe = df_tau[df_tau["region"].isin(df_lobe["Region name"])]

    fig, ax = plt.subplots(figsize=(20, 5))
    plot_conds_regs(
        ax,
        df_tau_lobe,
        reg_order=df_lobe["Region name"].tolist(),
        conds_order=["W", "N3", "R"],
        show_lines=False,
    )
    ax.set_title(lobe)
    # ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    fig.savefig(
        data_path.joinpath(fig_name + "_" + lobe + ".pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

# Plot one figure per stage
for cond in df_tau["cond"].unique():
    df_tau_cond = df_tau[df_tau["cond"] == cond]

    fig, ax = plt.subplots(figsize=(20, 5))
    plot_conds_regs(
        ax,
        df_tau_cond,
        reg_order=df_regions["Region name"].tolist(),
        conds_order=[cond],
        show_lines=False,
        show_lines_means=False
    )
    ax.set_title(cond)
    # ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    fig.savefig(
        data_path.joinpath(fig_name + "_stage_" + cond + ".pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

# Finally, plot all regions together
fig, ax = plt.subplots(figsize=(40, 5))
plot_conds_regs(
    ax,
    df_tau,
    reg_order=df_regions["Region name"].tolist(),
    conds_order=["W", "N3", "R"],
    show_lines=False,
    sep=0.5,
)
ax.set_title("All regions")
# ax.set_ylim(ylim)
ax.set_yscale(yscale)
fig.savefig(
    data_path.joinpath(fig_name + "_all_regs.pdf"),
    format="pdf",
    bbox_inches="tight",
)
plt.close(fig)
