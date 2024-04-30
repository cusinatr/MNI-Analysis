from pathlib import Path
from datetime import datetime
import pandas as pd
from mnitimescales import Load, Parcel, FitAcfSw

pd.options.mode.chained_assignment = None  # suppress pandas' warnings


class PipeTCSW:

    def __init__(
        self,
        mat_path: Path,
        sw_path: Path,
        results_path: Path,
        config_path: Path,
        parc_path: Path,
        stages=["N2", "N3"],
    ):

        self.mat_path = mat_path
        self.sw_path = sw_path
        self.results_path = results_path
        self.config_path = config_path
        self.parc_path = parc_path
        self.stages = stages

    def _save_results(self, df: pd.DataFrame, save_path: Path, save_name: str):

        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path.joinpath(save_name + ".csv"))

    def _log_pipe(
        self,
        epo_sws: float,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
    ):

        # Create log path
        log_path = self.results_path.joinpath("log.txt")
        with open(log_path, "w") as f:
            f.write("\n------------------------------------\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Timescales analysis on {self.stages} stages.\n")
            f.write(f"Results saved in {self.results_path}\n")
            f.write(f"Analysis parameters:\n")
            f.write(f"Epochs around slow waves: +-{epo_sws} s\n")
            f.write(f"Epoch duration: {epo_dur} s\n")
            f.write(f"Epoch overlap: {epo_overlap} s\n")
            f.write(f"Filtering: {filt}\n")
            f.write(f"Filtering frequencies: {filt_freqs} Hz\n")
            f.write(f"Number of lags: {nlags}\n")
            f.write(f"Timescales computation mode: {tau_mode}\n")
            f.write(f"Fit function: {fit_func}\n")
            f.write(f"Fit range: {fit_range} s\n")
            f.write("\n------------------------------------\n")

    def run(
        self,
        epo_sws: float,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
    ):

        # Log analysis
        self._log_pipe(
            epo_sws,
            epo_dur,
            epo_overlap,
            filt,
            filt_freqs,
            nlags,
            tau_mode,
            fit_func,
            fit_range,
        )

        # Load info dataframe & stage data
        load = Load(mat_path=self.mat_path)
        df_info = load.get_info()

        # Compute measure for each stage
        for stage in self.stages:

            raw_stage = load.load_raw_stage(stage, filt=filt, filt_freqs=filt_freqs)

            fit_acf_sw = FitAcfSw(
                df_info,
                raw_stage,
                stage,
                self.sw_path,
                self.results_path,
                self.config_path,
            )
            df_timescales_sw, df_acf_sw = fit_acf_sw.compute_timescales_sliding(
                epo_sws,
                epo_dur,
                epo_overlap,
                nlags,
                tau_mode,
                fit_func,
                fit_range,
            )

            # Save results
            self._save_results(
                df_timescales_sw, self.results_path, f"timescales_sw_{stage}"
            )
            self._save_results(df_acf_sw, self.results_path, f"acf_sw_{stage}")

            # 3) Parcellate results & save
            parc = Parcel(parc_path=self.parc_path)
            print("Parcellating in MNI atlas...")
            df_timescales_sw_mni = parc.parcel_mni(df_timescales_sw)
            self._save_results(
                df_timescales_sw_mni,
                self.results_path,
                f"timescales_sw_{stage}_mni",
            )
            for cond in df_timescales_sw.columns[10:].to_list():
                print(f"Parcellating in HCP-MMP atlas {cond} condition...")
                df_timescales_sw_mmp, df_timescales_sw_mmp_macro = parc.parcel_mmp(df_timescales_sw, cond)
                self._save_results(
                    df_timescales_sw_mmp,
                    self.results_path,
                    f"timescales_sw_{cond}_{stage}_mmp",
                )
                self._save_results(
                    df_timescales_sw_mmp_macro,
                    self.results_path,
                    f"timescales_sw_{cond}_{stage}_mmp_macro",
                )

