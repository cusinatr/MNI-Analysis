from pathlib import Path
import pandas as pd
import numpy as np


def create_pat_folder(pat_id: str, res_path: Path, chans: list):
    """Create the necassary folder structure to use ACF / PSD classes.

    Args:
        pat_id (str): patient ID
        res_path (Path): path where results are saved
        chans (list): channel names

    Returns:
        None
    """

    # Create folders & files
    pat_path = res_path.joinpath("Pats", pat_id)
    pat_path.mkdir(parents=True, exist_ok=True)
    proc_path = pat_path.joinpath("ProcData")
    proc_path.mkdir(parents=True, exist_ok=True)

    # Create info file
    df_meta_pat = pd.DataFrame(index=chans, columns=["lead", "bad"])
    df_meta_pat["lead"] = "depth"
    df_meta_pat["bad"] = False
    df_meta_pat.to_csv(pat_path.joinpath(pat_id + "_meta.csv"))

    return None


def convert_knee_tau(el_data: pd.Series) -> float:
    """Get timescale from knee fit, in milliseconds."""
    # Get knee and exponent
    knee = el_data["knee"]
    exp = el_data["exp"]

    # Knee frequency
    knee_freq = knee ** (1 / exp)

    return 1000 / (2 * np.pi * knee_freq)  # 1000 to convert to ms
