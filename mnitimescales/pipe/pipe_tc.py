from pathlib import Path
import pandas as pd
from mnitimescales import Load, FitACF, Parcel


class PipeTC:

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
        df.to_csv(save_path.joinpath(save_name + ".csv"), index=False)

    def run(
        self,
        epo_dur: float,
        epo_overlap: float,
        filt: bool,
        filt_freqs: list,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
    ):

        # Load info dataframe
        load = Load(mat_path=self.mat_path)
        df_info = load.get_info()

        # List for results across stages
        df_timescales_stages = []

        # Compute measure for each stage
        for stage in self.stages:

            # 1) Load data
            epo_stage = load.load_epo_stage(
                stage, epo_dur, epo_overlap, filt, filt_freqs
            )

            # 2) Fit ACF
            fit_acf = FitACF(
                df_info=df_info,
                epochs=epo_stage,
                stage=stage,
                results_path=self.results_path,
                config_path=self.config_path,
            )
            df_timescales = fit_acf.compute_timescales(
                nlags,
                tau_mode,
                fit_func,
                fit_range,
            )
            df_timescales_stages.append(df_timescales)

            # 3) Parcellate results
            parc = Parcel(parc_path=self.parc_path)
            df_timescales_parc, df_timescales_macro_mmp = parc.parcel_mmp(
                df_timescales, "tau"
            )
            df_timescales_mni = parc.parcel_mni(df_timescales)

            # 4) Save parcellated results
            self._save_results(
                df_timescales_parc,
                self.results_path,
                f"tau_parc_{stage}",
            )
            self._save_results(
                df_timescales_macro_mmp,
                self.results_path,
                f"tau_macro_mmp_{stage}",
            )
            self._save_results(
                df_timescales_mni,
                self.results_path,
                f"tau_macro_mni_{stage}",
            )

        # 5) Save results across stages
        df_timescales_stages = pd.concat(df_timescales_stages, ignore_index=True)
        self._save_results(
            df_timescales_stages,
            self.results_path,
            "tau_stages",
        )
