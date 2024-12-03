from pathlib import Path
import pandas as pd
from ieeganalysis import ACF, TAU
from mnitimescales.utils import create_res_df
from .utils import create_pat_folder


class FitACF:
    """Fit ACFs from epoched data.

    Args:
        df_info (pd.DataFrame): metadata for each channel in the dataset.
        epochs (dict): keys are pat names, values MNE Epochs.
        stage (str): sleep stage being analyzed.
        results_path (str): path where to store results.
        config_path (str): path to file with analysis configurations.
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
        self.nlags = None
        self.tau_mode = None
        self.fit_func = None
        self.fit_range = None

    def _compute_timescales_pat(self, pat_id, chans_pat, plot=True):

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
        if plot:
            patTAU.plot_timescales()

        return list(tau_pat.values())

    def compute_timescales(
        self,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
        plot=True
    ) -> pd.DataFrame:
        """Compute timescales per channels from a sample of ACFs.
        Timescales are computed in the average ACF.

        Args:
            nlags (int): number of lags to compute for the ACF.
            tau_mode (str): Timescales computation modality. Can be 'fit' or 'interp'.
            fit_func (str): function to use for fitting.
            fit_range (list): range of fitting in s.
            plot (bool, optional): whether to plot results for each patient. Defaults to True.

        Returns:
            pd.DataFrame: df_info with added timescales values.
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
            timescales_pat = self._compute_timescales_pat(pat, chans_pat, plot)
            df_timescales_pat = create_res_df(
                df_info_pat, self.epochs[pat].ch_names, self.stage, columns_res=["tau"]
            )
            df_timescales_pat["tau"] = timescales_pat
            df_timescales.append(df_timescales_pat)

        # Concatenate results
        df_timescales = pd.concat(df_timescales, ignore_index=True)

        return df_timescales
