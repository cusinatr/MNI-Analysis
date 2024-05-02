"""
Module for computation of spatial correlations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from . import utils


class SC:

    def __init__(
        self,
        df_info: pd.DataFrame,
        df_regions: pd.DataFrame,
        epochs: dict,
        stage: str,
        results_path: Path,
    ):

        self.df_info = df_info
        self.df_regions = df_regions
        self.epochs = epochs
        self.stage = stage
        self.results_path = results_path
        # For analysis
        self.df_sc = None
        self.freq_band = None
        self.use_bispectrum = None

    def compute_sc(
        self,
        freq_band=None,
        use_bispectrum=False,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            fit_mode (str): _description_
            fit_range (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        self.freq_band = freq_band
        self.use_bispectrum = use_bispectrum

        pats = self.df_info["pat"].unique().tolist()
        df_sc = []

        for pat in pats:
            print("Patient: ", pat)

            df_info_pat = self.df_info[self.df_info["pat"] == pat]

            # Check epochs are available
            if self.epochs[pat] is None:
                continue
            df_sc_pat = utils.compute_cc(
                self.epochs[pat],
                df_info_pat,
                self.df_regions,
                use_bispectrum,
                freq_band,
            )
            df_sc_pat.insert(0, "pat", pat)
            df_sc.append(df_sc_pat)

        # Concatenate results
        df_sc = pd.concat(df_sc, ignore_index=True)

        return df_sc

    def fit_sc(self, df_sc: pd.DataFrame, fit_mode: str, col_name: str):

        if fit_mode == "fit":
            # Check type of fit
            if col_name == "corr":
                fit_type = "exp"
            elif col_name == "lag":
                fit_type = "lin"
            # Check bounds values
            if self.use_bispectrum:
                upper_bounds = (100, np.inf, np.inf)
            else:
                upper_bounds = (100, 1, 1)
            df_sc_fit = utils.exp_sc(df_sc, col_name, fit_type, upper_bounds)
        elif fit_mode == "median":
            df_sc_fit = utils.median_sc(df_sc, col_name)
        elif fit_mode == "auc":
            df_sc_fit = utils.auc_sc(df_sc, col_name)
        else:
            raise ValueError("fit_mode must be one of 'fit', 'median', 'auc'")

        return df_sc_fit
