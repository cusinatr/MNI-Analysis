from pathlib import Path
import pandas as pd
import numpy as np
from ieeganalysis import PSD, SpectralParam
from mnitimescales.utils import create_res_df
from .utils import create_pat_folder, convert_knee_tau


class FitPSD:
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
        results_path: Path,
        config_path: Path,
    ):

        self.df_info = df_info
        self.epochs = epochs
        self.stage = stage
        self.results_path = results_path
        self.config_path = config_path
        # Parameters for analysis
        self.fit_mode = None
        self.fit_range = None

    def _compute_psd_pat(self, pat_id, chans_pat):

        # Create folders & files
        create_pat_folder(pat_id, self.results_path, chans_pat)

        # Prepare data analyses
        patPSD = PSD(self.config_path, pat_id)
        patSP = SpectralParam(self.config_path, pat_id)
        # Compute PSD
        patPSD.prepare_data(epochs=self.epochs[pat_id], stage=self.stage)
        psd_pat = patPSD.compute_psd(f_band=[1, 80], label="")
        # Compute Parametrization
        patSP.prepare_data(psd_pat, stage=self.stage, label=self.stage)
        fg_pat = patSP.parametrize_psd(
            frange=self.fit_range, aperiodic_mode=self.fit_mode, save_full=False
        )
        patSP.plot_parametrization(plot_range=[1, 80])
        exp_pat = fg_pat.get_params("aperiodic_params", "exponent")
        r2_pat = fg_pat.get_params("r_squared")
        if self.fit_mode == "knee":
            tau_pat = convert_knee_tau(
                {
                    "knee": fg_pat.get_params("aperiodic_params", "knee"),
                    "exp": exp_pat,
                }
            )
        else:
            tau_pat = np.array([np.nan] * len(fg_pat))

        return pd.DataFrame(
            {"r2": r2_pat, "exp": exp_pat, "tau": tau_pat}, index=chans_pat
        )

    def compute_psd(
        self,
        fit_mode: str,
        fit_range: list,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            fit_mode (str): _description_
            fit_range (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        self.fit_mode = fit_mode
        self.fit_range = fit_range

        pats = self.df_info["pat"].unique().tolist()
        df_psd = []

        for pat in pats:
            print("Patient: ", pat)

            df_info_pat = self.df_info[self.df_info["pat"] == pat]
            chans_pat = df_info_pat["chan"].to_list()

            # Check epochs are available
            if self.epochs[pat] is None:
                continue
            psd_pat = self._compute_psd_pat(pat, chans_pat)
            df_psd_pat = create_res_df(
                df_info_pat,
                self.epochs[pat].ch_names,
                self.stage,
                columns_res=["r2", "exp", "tau"],
            )
            df_psd_pat[["r2", "exp", "tau"]] = psd_pat.loc[
                df_psd_pat["chan"]
            ].to_numpy()
            df_psd.append(df_psd_pat)

        # Concatenate results
        df_psd = pd.concat(df_psd, ignore_index=True)

        return df_psd
