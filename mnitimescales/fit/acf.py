from pathlib import Path
import pandas as pd
from ieeganalysis import ACF, TAU
from .utils import create_pat_folder, create_res_df


class FitACF:
    """_summary_

    Args:
        df_info (pd.DataFrame): _description_
        epochs (dict): _description_
        stage (str): _description_
        results_path (str): _description_
        config_path (str): _description_
    """

    def __init__(
        self,
        df_info: pd.DataFrame,
        epochs: dict,
        stage: str,
        results_path: str,
        config_path: str,
    ):

        self.df_info = df_info
        self.epochs = epochs
        self.stage = stage
        self.results_path = Path(results_path)
        self.config_path = Path(config_path)
        # Parameters for analysis
        self.nlags = None
        self.tau_mode = None
        self.fit_func = None
        self.fit_range = None

    def _compute_timescales_pat(self, pat_id, chans_pat):

        # Create folders & files
        create_pat_folder(pat_id, self.results_path, chans_pat)

        # Prepare data analyses
        patACF = ACF(self.config_path, pat_id)
        patTAU = TAU(self.config_path, pat_id)
        # Compute ACF
        patACF.prepare_data(epochs=self.epochs[pat_id], stage=self.stage)
        acf_pat = patACF.compute_acf(nlags=self.nlags, label="")
        # Compute Timescale
        patTAU.prepare_data(acf_pat, stage=self.stage, label=self.stage)
        tau_pat = patTAU.compute_timescales(
            mode=self.tau_mode, fit_func=self.fit_func, fit_range=self.fit_range
        )
        patTAU.plot_timescales()

        return list(tau_pat.values())

    def compute_timescales(
        self,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            nlags (int): _description_
            tau_mode (str): _description_
            fit_func (str): _description_
            fit_range (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        self.nlags = nlags
        self.tau_mode = tau_mode
        self.fit_func = fit_func
        self.fit_range = fit_range

        pats = self.df_info["pat"].unique().tolist()
        df_timescales = []

        for pat in pats:
            print("Patient: ", pat)

            df_info_pat = self.df_info[self.df_info["pat"] == pat]
            chans_pat = df_info_pat["chan"].to_list()

            # Check epochs are available
            if self.epochs[pat] is None:
                continue
            timescales_pat = self._compute_timescales_pat(pat, chans_pat)
            df_timescales_pat = create_res_df(
                df_info_pat, self.epochs[pat].ch_names, self.stage, columns_res=["tau"]
            )
            df_timescales_pat["tau"] = timescales_pat
            df_timescales.append(df_timescales_pat)

        # Concatenate results
        df_timescales = pd.concat(df_timescales, ignore_index=True)

        return df_timescales
