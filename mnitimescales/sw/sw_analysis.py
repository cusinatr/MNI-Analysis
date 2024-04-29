from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import mne
from . import utils as utils_sw
from . import plots as plots_sw
from mnitimescales.utils import create_res_df


class ComputeSW:

    def __init__(
        self,
        df_info: pd.DataFrame,
        raws: dict,
        stage: str,
        results_path: Path,
        sw_freqs=[0.5, 4],
        gamma_freqs=[30, 80],
    ):

        self.df_info = df_info
        self.raws = raws
        self.stage = stage
        self.results_path = results_path
        self.sw_path = self.results_path.joinpath("Pats")
        self.sw_path.mkdir(exist_ok=True)
        # Parameters for analysis
        self.sw_freqs = sw_freqs
        self.gamma_freqs = gamma_freqs
        self.dur_threshold = None
        self.dur_neg = None
        self.dur_pos = None
        self.amp_percentile = None
        self.center_sws = None
        self.t_epo_sws = None

    def _prepare_raw(self, raw):

        raw_ds = utils_sw.downsample_raw(raw)
        hypno = utils_sw.load_hypnogram(raw_ds, self.stage)
        # Low frequency raw
        raw_sw = raw_ds.copy().filter(*self.sw_freqs, verbose=False)
        # Gamma power raw
        raw_gamma = raw.copy().filter(*self.gamma_freqs, verbose=False)
        raw_gamma = raw_gamma.apply_hilbert(envelope=True)
        raw_gamma._data = 2 * np.log10(raw_gamma._data)  # get log power

        return raw_sw, raw_gamma, hypno

    def _detect_sw_pat(self, pat):

        raw_sw, raw_gamma, hypno = self._prepare_raw(self.raws[pat])

        sw_events, _, epo_sw, epo_gamma = utils_sw.detect_sws_gamma(
            raw_sw,
            raw_gamma,
            hypno,
            stages=(2, 3),
            dur_threshold=self.dur_threshold,
            dur_neg=self.dur_neg,
            dur_pos=self.dur_pos,
            use_percentile=True,
            amp_percentile=self.amp_percentile,
            center_sws=self.center_sws,
            t_epoch_sws=self.t_epo_sws,
        )

        # Compute SW overlap
        sw_overlap = utils_sw.sw_conn(sw_events)
        # Theshold of "involvement" between local and global SWs
        loc_glo_threshold, sw_glo_bool = utils_sw.compute_sw_global_threshold(
            sw_overlap
        )
        # Add column in events dataframe
        sw_glo_bool = np.concatenate(
            list({ch: sw_glo_bool[ch] for ch in sw_events.Channel.unique()}.values()),
            dtype=int,
        )
        sw_events.insert(11, "Global", sw_glo_bool)

        # Compte SW density
        sw_density = utils_sw.sw_density(
            sw_events,
            hypno,
            raw_sw.info["ch_names"],
            raw_sw.info["sfreq"],
        )

        # Plot results
        plots_sw.plot_sw_gamma(
            epo_sw,
            epo_gamma,
            t_epoch_sws=self.t_epo_sws,
            save_path=self.pat_path,
            save_name=f"sw_gamma_{self.stage}",
        )
        plots_sw.plot_sw_loc_glo(
            epo_sw,
            sw_overlap,
            loc_glo_threshold,
            self.t_epo_sws,
            save_path=self.pat_path,
            save_name=f"sw_loc_glo_{self.stage}",
        )
        plots_sw.plot_sw_overlap(
            sw_overlap, save_path=self.pat_path, save_name=f"sw_overlap_{self.stage}"
        )

        return sw_events, sw_density, sw_overlap, epo_sw, epo_gamma

    def detect_sw(
        self,
        dur_threshold=(0.5, 2),
        dur_neg=(0.1, 2),
        dur_pos=(0.1, 2),
        amp_percentile=25,
        center_sws="NegPeak",
        t_epo_sws=2,
    ):

        self.dur_threshold = dur_threshold
        self.dur_neg = dur_neg
        self.dur_pos = dur_pos
        self.amp_percentile = amp_percentile
        self.center_sws = center_sws
        self.t_epo_sws = t_epo_sws

        pats = self.df_info["pat"].unique().tolist()
        df_density = []
        epochs_sw = {}
        epochs_gamma = {}

        for pat in pats:
            print("Patient: ", pat)
            self.pat_path = self.sw_path.joinpath(pat)
            self.pat_path.mkdir(exist_ok=True)
            df_info_pat = self.df_info[self.df_info["pat"] == pat]

            # Check raw is available
            if self.raws[pat] is None:
                continue
            sw_events_pat, sw_density_pat, sw_overlap_pat, epo_sw_pat, epo_gamma_pat = (
                self._detect_sw_pat(pat)
            )
            df_density_pat = create_res_df(
                df_info_pat,
                self.raws[pat].ch_names,
                self.stage,
                columns_res=["total", "local", "global"],
            )
            df_density_pat[["total", "local", "global"]] = sw_density_pat.loc[
                df_density_pat["chan"]
            ].to_numpy()
            df_density.append(df_density_pat)

            # Add epochs
            epochs_sw[pat] = epo_sw_pat
            epochs_gamma[pat] = epo_gamma_pat

            # Save results
            sw_events_pat.to_csv(self.pat_path.joinpath(f"sw_events_{self.stage}.csv"))
            sw_density_pat.to_csv(
                self.pat_path.joinpath(f"sw_density_{self.stage}.csv")
            )
            with open(
                self.pat_path.joinpath(f"sw_overlap_{self.stage}.pkl"), "wb"
            ) as f:
                pickle.dump(sw_overlap_pat, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Concatenate results
        df_density = pd.concat(df_density, ignore_index=True)

        # Finally, save epochs
        with open(self.results_path.joinpath(f"epochs_sw_{self.stage}.pkl"), "wb") as f:
            pickle.dump(epochs_sw, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            self.results_path.joinpath(f"epochs_gamma_{self.stage}.pkl"), "wb"
        ) as f:
            pickle.dump(epochs_gamma, f, protocol=pickle.HIGHEST_PROTOCOL)

        return df_density
