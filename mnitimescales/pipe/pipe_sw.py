from pathlib import Path
from datetime import datetime
import pandas as pd
from mnitimescales import Load, ComputeSW, Parcel
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress yasa's warnings
pd.options.mode.chained_assignment = None  # suppress pandas' warnings


class PipeSW:

    def __init__(
        self,
        mat_path: Path,
        results_path: str,
        parc_path: Path,
        stages=["N2", "N3"],
    ):

        self.mat_path = mat_path
        self.results_path = results_path
        self.parc_path = parc_path
        self.stages = stages

    def _save_results(self, df: pd.DataFrame, save_path: Path, save_name: str):

        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path.joinpath(save_name + ".csv"))

    def _log_pipe(
        self,
        sw_freqs: tuple,
        gamma_freqs: tuple,
        dur_threshold: tuple,
        dur_neg: tuple,
        dur_pos: tuple,
        amp_percentile: float,
        center_sws: str,
        t_epo_sws: float,
    ):

        # Create log path
        log_path = self.results_path.joinpath("log.txt")
        with open(log_path, "w") as f:
            f.write("\n------------------------------------\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Slow waves analysis on {self.stages} stages.\n")
            f.write(f"Results saved in {self.results_path}\n")
            f.write(f"Analysis parameters:\n")
            f.write(f"SW frequencies: {sw_freqs} Hz\n")
            f.write(f"Gamma frequencies: {gamma_freqs} Hz\n")
            f.write(f"Duration threshold: {dur_threshold} s\n")
            f.write(f"Duration negative peak: {dur_neg} s\n")
            f.write(f"Duration positive peak: {dur_pos} s\n")
            f.write(f"Amplitude percentile criterion: {amp_percentile} %\n")
            f.write(f"Center for epochs: {center_sws}\n")
            f.write(f"Duration of epochs: +-{t_epo_sws} s\n")
            f.write("\n------------------------------------\n")

    def run(
        self,
        sw_freqs: tuple,
        gamma_freqs: tuple,
        dur_threshold: tuple,
        dur_neg: tuple,
        dur_pos: tuple,
        amp_percentile: float,
        center_sws: str,
        t_epo_sws: float,
    ):

        # Log analysis
        self._log_pipe(
            sw_freqs,
            gamma_freqs,
            dur_threshold,
            dur_neg,
            dur_pos,
            amp_percentile,
            center_sws,
            t_epo_sws,
        )

        # Load info dataframe
        load = Load(mat_path=self.mat_path)
        df_info = load.get_info()

        # List for results across stages
        df_density_stages = []

        # Compute measure for each stage
        for stage in self.stages:

            # 1) Load data
            raw_stage = load.load_raw_stage(stage)

            # 2) Compute SWs
            compute_sw = ComputeSW(
                df_info=df_info,
                raws=raw_stage,
                stage=stage,
                results_path=self.results_path,
                sw_freqs=sw_freqs,
                gamma_freqs=gamma_freqs,
            )
            df_density = compute_sw.detect_sw(
                dur_threshold=dur_threshold,
                dur_neg=dur_neg,
                dur_pos=dur_pos,
                amp_percentile=amp_percentile,
                center_sws=center_sws,
                t_epo_sws=t_epo_sws,
            )

            df_density_stages.append(df_density)

            # 3) Parcellate results & save
            parc = Parcel(parc_path=self.parc_path)
            print("Parcellating in MNI atlas...")
            df_density_mni = parc.parcel_mni(df_density)
            self._save_results(
                df_density_mni,
                self.results_path,
                f"density_{stage}_mni",
            )
            for cond in ["total", "local", "global"]:
                print(f"Parcellating in HCP-MMP atlas {cond} condition...")
                df_density_mmp, df_density_mmp_macro = parc.parcel_mmp(df_density, cond)
                self._save_results(
                    df_density_mmp,
                    self.results_path,
                    f"density_{cond}_{stage}_mmp",
                )
                self._save_results(
                    df_density_mmp_macro,
                    self.results_path,
                    f"density_{cond}_{stage}_mmp_macro",
                )

        # 4) Save results across stages
        df_density_stages = pd.concat(df_density_stages, ignore_index=True)
        self._save_results(
            df_density_stages,
            self.results_path,
            "density_stages",
        )
