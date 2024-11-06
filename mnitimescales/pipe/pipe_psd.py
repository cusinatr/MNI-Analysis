from pathlib import Path
from datetime import datetime
import pandas as pd
from mnitimescales import Load, FitPSD, Parcel


class PipePSD:
    """
    _summary_

    Args:
        mat_path (Path): path to the .mat file with the data.
        results_path (str): _description_
        config_path (str): _description_
        parc_path (Path): _description_
        stages (list, optional): _description_. Defaults to ["W", "N2", "N3", "R"].
    """

    def __init__(
        self,
        mat_path: Path,
        results_path: str,
        config_path: str,
        parc_path: Path,
        stages=["W", "N2", "N3", "R"],
    ):

        self.mat_path = mat_path
        self.results_path = results_path
        self.config_path = config_path
        self.parc_path = parc_path
        self.stages = stages

    def _save_results(self, df: pd.DataFrame, save_path: Path, save_name: str):

        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path.joinpath(save_name + ".csv"))

    def _log_pipe(
        self,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        fit_mode: str,
        fit_range: list,
    ):

        # Create log path
        log_path = self.results_path.joinpath("log.txt")
        with open(log_path, "w") as f:
            f.write("\n------------------------------------\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"PSD analysis on {self.stages} stages.\n")
            f.write(f"Results saved in {self.results_path}\n")
            f.write(f"Analysis parameters:\n")
            f.write(f"Epoch duration: {epo_dur}\n")
            f.write(f"Epoch overlap: {epo_overlap}\n")
            f.write(f"Filtering: {filt}\n")
            f.write(f"Filtering frequencies: {filt_freqs}\n")
            f.write(f"PSD parametrization mode: {fit_mode}\n")
            f.write(f"Fit range: {fit_range}\n")
            f.write("\n------------------------------------\n")

    def run(
        self,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        fit_mode: str,
        fit_range: list,
    ):

        # Log analysis
        self._log_pipe(
            epo_dur,
            epo_overlap,
            filt,
            filt_freqs,
            fit_mode,
            fit_range,
        )

        # Load info dataframe
        load = Load(mat_path=self.mat_path)
        df_info = load.get_info()

        # List for results across stages
        df_psd_stages = []

        # Compute measure for each stage
        for stage in self.stages:

            # 1) Load data
            epo_stage = load.load_epo_stage(
                stage, epo_dur, epo_overlap, filt, filt_freqs
            )

            # 2) Fit PSD
            fit_psd = FitPSD(
                df_info=df_info,
                epochs=epo_stage,
                stage=stage,
                results_path=self.results_path,
                config_path=self.config_path,
            )
            df_psd = fit_psd.compute_psd(
                fit_mode,
                fit_range,
            )
            df_psd_stages.append(df_psd)

            # 3) Parcellate results & save
            parc = Parcel(parc_path=self.parc_path)
            df_psd_mni = parc.parcel_mni(df_psd)
            df_psd_mmp, df_psd_mmp_macro = parc.parcel_mmp(df_psd, "exp")
            self._save_results(
                df_psd_mmp,
                self.results_path,
                f"exp_{stage}_mmp",
            )
            self._save_results(
                df_psd_mmp_macro,
                self.results_path,
                f"exp_{stage}_mmp_macro",
            )
            self._save_results(
                df_psd_mni,
                self.results_path,
                f"exp_{stage}_mni",
            )
            # Save also timescale if fit is knee
            if fit_mode == "knee":
                df_tau_mmp, df_tau_mmp_macro = parc.parcel_mmp(df_psd, "tau")
                self._save_results(
                    df_tau_mmp,
                    self.results_path,
                    f"tau_{stage}_mmp",
                )
                self._save_results(
                    df_tau_mmp_macro,
                    self.results_path,
                    f"tau_{stage}_mmp_macro",
                )

        # 5) Save results across stages
        df_psd_stages = pd.concat(df_psd_stages, ignore_index=True)
        self._save_results(
            df_psd_stages,
            self.results_path,
            "psd_stages",
        )
