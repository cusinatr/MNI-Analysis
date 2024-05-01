"""
Module for computation of spatial correlations.
"""
from pathlib import Path
import pandas as pd
from mnitimescales.utils import create_res_df
from . import utils


class SC():

    def __init__(
        self,
        df_info: pd.DataFrame,
        epochs: dict,
        stage: str,
        results_path: Path,
        config_path: Path,
    ):

        self.df_info = df_info
        self.epochs = epochs
        self.stage = stage
        self.results_path = results_path
        self.config_path = config_path
        # TODO: Parameters for analysis
        

    def _compute_sc_pat(self, pat_id, chans_pat):

        

        return

    def compute_sc(
        self,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            fit_mode (str): _description_
            fit_range (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        pats = self.df_info["pat"].unique().tolist()
        df_sc = []

        for pat in pats:
            print("Patient: ", pat)

            df_info_pat = self.df_info[self.df_info["pat"] == pat]
            chans_pat = df_info_pat["chan"].to_list()

            # Check epochs are available
            if self.epochs[pat] is None:
                continue
            df_sc_pat = self._compute_sc_pat(pat, chans_pat)
            # df_sc_pat = create_res_df(
            #     df_info_pat,
            #     self.epochs[pat].ch_names,
            #     self.stage,
            #     columns_res=["r2", "exp", "tau"],
            # )
            # df_sc_pat[["r2", "exp", "tau"]] = sc_pat.loc[
            #     df_sc_pat["chan"]
            # ].to_numpy()
            df_sc.append(df_sc_pat)

        # Concatenate results
        df_sc = pd.concat(df_sc, ignore_index=True)

        return df_sc