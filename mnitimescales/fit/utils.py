from pathlib import Path
import pandas as pd


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


def create_res_df(
    df_info_pat: pd.DataFrame,
    stage: str,
    columns_res=[],
):
    """Create a dataframe for results of a patient.

    Args:
        df_info_pat (pd.DataFrame): metadata about patient channels.
        pat_id (str): ID for patient.
        age (float): age of patient.
        gender (str): gender of patient.
        stage (str): sleep stage.
        columns_res (list): column names to add for results.

    Returns:
        pd.DataFrame: dataframe with metadata for patient and empty results column(s).
    """

    df_res_pat = pd.DataFrame(
        columns=[
            "pat",
            "age",
            "gender",
            "chan",
            "type",
            "stage",
            "region",
            "mni_x",
            "mni_y",
            "mni_z",
        ]
        + columns_res
    )
    df_res_pat["pat"] = df_info_pat["pat"].to_list()
    df_res_pat["age"] = df_info_pat["age"].to_list()
    df_res_pat["gender"] = df_info_pat["gender"].to_list()
    df_res_pat["chan"] = df_info_pat["chan"].to_list()
    df_res_pat["type"] = df_info_pat["type"].to_list()
    df_res_pat["stage"] = [stage] * len(df_info_pat)
    df_res_pat["region"] = df_info_pat["region"].to_list()
    df_res_pat["mni_x"] = df_info_pat["mni_x"].to_list()
    df_res_pat["mni_y"] = df_info_pat["mni_y"].to_list()
    df_res_pat["mni_z"] = df_info_pat["mni_z"].to_list()

    return df_res_pat
