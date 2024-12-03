from pathlib import Path
from datetime import datetime
import pandas as pd
from mnitimescales import Load, FitPSD, Parcel


class PipePSD:
    """
    Pipeline to run 'PSD parametrization' analysis for different sleep stages.
    1. Create epochs with desired duration and overlap.
    2. Filter data and extract power (optional).
    2. Fit PSD and extract parameter.
    3. Parcellate results into a surface atlas (HCPMMP1 supported)

    Args:
        mat_path (Path): path to the .mat file with the MNI Atlas data.
        results_path (Path): path where to save results.
        config_path (Path): path to yaml configuration file.
        parc_path (Path): path with parcellation files (regions coordinates and .nii).
        stages (list, optional): sleep stags to analyze. Defaults to ["W", "N2", "N3", "R"].
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
        plot=True
    ):
        """Function to actually run the pipeline.

        Args:
            epo_dur (float): epoch duration in s.
            epo_overlap (float): epoch overlap in s.
            filt (bool): Whether to filter data.
            filt_freqs (list): (low, high) frequencies for the filter.
            fit_mode (str): PSD fitting mode, either 'fixed' or 'knee'.
            fit_range (list): frequency range of PSD fitting.
            plot (bool, optional): whether to plot results for each patient. Defaults to True.
        """

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
                plot
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
