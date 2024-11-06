from pathlib import Path
from datetime import datetime
import pandas as pd
from mnitimescales import Load, SC


class PipeSC:

    def __init__(
        self,
        mat_path: Path,
        results_path: str,
        df_regions: pd.DataFrame,
        stages=["W", "N2", "N3", "R"],
    ):

        self.mat_path = mat_path
        self.results_path = results_path
        self.df_regions = df_regions
        self.stages = stages
        self.df_info = None

    def _save_results(self, df: pd.DataFrame, save_path: Path, save_name: str):

        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path.joinpath(save_name + ".csv"))

    def _log_compute(
        self,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        use_bispectrum: bool,
    ):

        # Create log path
        log_path = self.results_path.joinpath("log.txt")
        with open(log_path, "w") as f:
            f.write("\n------------------------------------\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Spatial correlations analysis on {self.stages} stages.\n")
            f.write(f"Results saved in {self.results_path}\n")
            f.write(f"Computation parameters:\n")
            f.write(f"Epoch duration: {epo_dur}\n")
            f.write(f"Epoch overlap: {epo_overlap}\n")
            f.write(f"Filtering: {filt}\n")
            f.write(f"Filtering frequencies: {filt_freqs}\n")
            f.write(f"Use bispectrum: {use_bispectrum}\n")
            f.write("\n------------------------------------\n")

    def _log_fit(
        self,
        fit_mode: str,
        col_name: str,
    ):

        # Create log path
        log_path = self.results_path.joinpath("log.txt")
        with open(log_path, "a") as f:
            f.write("\n------------------------------------\n")
            f.write(f"Fit parameters:\n")
            f.write(f"Fit mode: {fit_mode}\n")
            f.write(f"Analysis variable : {col_name}\n")
            f.write("\n------------------------------------\n")

    def run_compute(
        self,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        use_bispectrum: bool,
    ):

        # Log analysis
        self._log_compute(
            epo_dur,
            epo_overlap,
            filt,
            filt_freqs,
            use_bispectrum,
        )

        # Load info dataframe
        load = Load(mat_path=self.mat_path)
        self.df_info = load.get_info()

        # List for results across stages
        df_sc_stages = []

        # Compute measure for each stage
        for stage in self.stages:

            # 1) Load data
            if use_bispectrum:
                epo_stage = load.load_epo_stage(
                    stage, epo_dur, epo_overlap, filt=False
                )
            else:
                epo_stage = load.load_epo_stage(
                    stage, epo_dur, epo_overlap, filt, filt_freqs
                )

            # 2) Compute sc
            sc = SC(
                df_info=self.df_info,
                df_regions=self.df_regions,
                epochs=epo_stage,
                stage=stage,
                results_path=self.results_path,
            )
            if filt:
                freq_band_sc = filt_freqs
            else:
                freq_band_sc = None
            df_sc = sc.compute_sc(
                use_bispectrum=use_bispectrum,
                freq_band=freq_band_sc,
            )
            df_sc.insert(1, "stage", stage)
            df_sc_stages.append(df_sc)

            # Save stage results
            self._save_results(
                df_sc,
                self.results_path,
                f"sc_{stage}",
            )

        # Save results across stages
        self.df_sc_stages = pd.concat(df_sc_stages, ignore_index=True)
        self._save_results(
            self.df_sc_stages,
            self.results_path,
            "sc_all_stages",
        )

    def run_fit(self, fit_modes: list, col_names: list):

        for fit_mode, col_name in zip(fit_modes, col_names):
            # Log analysis
            self._log_fit(
                fit_mode,
                col_name,
            )

            # Compute measure for each stage
            for stage in self.stages:

                df_sc_stage = self.df_sc_stages[self.df_sc_stages["stage"] == stage]
                sc = SC(
                    df_info=self.df_info,
                    df_regions=self.df_regions,
                    epochs=[],
                    stage=stage,
                    results_path=self.results_path,
                )
                df_sc_params = sc.fit_sc(df_sc_stage, fit_mode, col_name)

                # Save stage results
                self._save_results(
                    df_sc_params,
                    self.results_path,
                    f"sc_params_{stage}_{fit_mode}_{col_name}",
                )
