from pathlib import Path
import pickle
import pandas as pd
from . import utils as utils_sw
from . import plots as plots_sw


class ComputeSW:

    def __init__(
        self,
        df_info: pd.DataFrame,
        raws: dict,
        stage: str,
        results_path: str,
        sw_freqs=[0.5, 4],
        gamma_freqs=[40, 80],
    ):

        self.df_info = df_info
        self.raws = raws
        self.stage = stage
        self.results_path = results_path
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
        hypno = utils_sw.load_hypnogram(raw_ds)  # TODO: crete surrogate hypno

        # TODO: create raw object
        # Low frequency
        raw_sw = raw_ds.copy().filter(
            0.3,
            1.25,
            # l_trans_bandwidth=0.3,
            # h_trans_bandwidth=1.6,
        )

        # Gamma band
        raw_gamma = raw.copy().filter(
            40,
            80,
        )
        raw_gamma = raw_gamma.apply_hilbert(envelope=True)

        return raw_sw, raw_gamma, hypno

    def _detect_sw_pat(self, pat, chans_pat):

        raw_sw, raw_gamma, hypno = self._prepare_raw(self.raws[pat])

        sw_events, _, epo_swa, epo_gamma = utils_sw.detect_sws_gamma(
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
        sw_overlap, sw_delays = utils_sw.sw_conn(sw_events, sw_window=0.3)

        # Save rerults
        sw_density = utils_sw.sw_density(
            sw_events,
            hypno,
            raw_sw.info["ch_names"],
            raw_sw.info["sfreq"],
        )
        sw_density.to_csv(self.results_path.joinpath("SW_density.csv"))
        sw_events.to_csv(self.results_path.joinpath("SW_events.csv"))
        with open(self.results_path.joinpath("SW_overlap.pkl"), "wb") as f:
            pickle.dump(sw_overlap, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.results_path.joinpath("SW_delays.pkl"), "wb") as f:
            pickle.dump(sw_delays, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot results
        plots_sw.plot_sw_gamma(
            epo_swa,
            epo_gamma,
            t_epoch_sws=self.t_epo_sws,
            save_path=self.results_path,
            save_name="SWA_gamma",
        )
        # Threshold for detecting global waves
        plots_sw.plot_sw_loc_glo(
            epo_swa, sw_overlap, self.t_epo_sws, save_path=self.results_path
        )
        plots_sw.plot_sw_overlap(sw_overlap, sw_delays, save_path=self.results_path)

    def detect_sw(
        self,
        dur_threshold=(0.8, 2),
        dur_neg=(0.25, 1.0),
        dur_pos=(0.25, 1.0),
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
        for pat in pats:
            print("Patient: ", pat)
            df_info_pat = self.df_info[self.df_info["pat"] == pat]
            chans_pat = df_info_pat["chan"].to_list()
            # Check raw is available
            if self.raws[pat] is None:
                continue
            self._detect_sw_pat(pat, chans_pat)
