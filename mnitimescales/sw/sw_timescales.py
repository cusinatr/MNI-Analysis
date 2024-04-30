from pathlib import Path
import numpy as np
import pandas as pd
import mne
from ieeganalysis import ACF, TAU
from . import utils as utils_sw
import mnitimescales.fit.utils as utils_acf
from mnitimescales.utils import create_res_df


class FitAcfSw:
    """_summary_

    Args:
        df_info (pd.DataFrame): _description_
        raws (dict): _description_
        stage (str): _description_
        results_path (str): _description_
        config_path (str): _description_
    """

    def __init__(
        self,
        df_info: pd.DataFrame,
        raws: dict,
        stage: str,
        sw_path: Path,
        results_path: Path,
        config_path: Path,
    ):

        self.df_info = df_info
        self.raws = raws
        self.stage = stage
        self.sw_path = sw_path
        self.results_path = results_path
        self.config_path = config_path
        # Parameters for analysis
        self.epo_sws = None
        self.epo_dur = None
        self.epo_overlap = None
        self.nlags = None
        self.tau_mode = None
        self.fit_func = None
        self.fit_range = None

    def _compute_timescales_chan(
        self, epo_chan: np.ndarray, pat: str, chan: str, sfreq: float, patACF, patTAU
    ):

        df_timescales_sw_chan = []
        df_acf_sw_chan = []     

        t_epo = np.linspace(-self.epo_sws, self.epo_sws, epo_chan.shape[1])
        info_chan = mne.create_info([chan], sfreq, ch_types="seeg")

        for t_start in np.arange(-self.epo_sws, self.epo_sws, self.epo_overlap):
            t_end = t_start + self.epo_dur
            if t_end >= self.epo_sws:
                continue
            epo_chan_sw = epo_chan[:, (t_epo > t_start) & (t_epo <= t_end)]
            epo_chan_sw = mne.EpochsArray(
                np.expand_dims(epo_chan_sw, axis=1),
                info_chan,
                tmin=t_start,
                verbose=False,
            )
            # Compute ACF
            patACF.prepare_data(epochs=epo_chan_sw)
            acf_pat = patACF.compute_acf(
                nlags=self.nlags, label=f"{chan}_{(t_start + t_end) / 2}", verbose=False
            )
            # Compute Timescale
            patTAU.prepare_data(acf_pat, label=f"{chan}_{(t_start + t_end) / 2}")
            tau_pat = patTAU.compute_timescales(
                mode=self.tau_mode,
                fit_func=self.fit_func,
                fit_range=self.fit_range,
                verbose=False,
            )

            df_timescales_sw_chan.append(pd.DataFrame(
                {
                    "pat": pat,
                    "chan": chan,
                    "time": (t_start + t_end) / 2,
                    "tau": tau_pat[chan],
                },
                index=[0],
            ))
            df_acf_sw_chan.append(pd.DataFrame(
                {
                    "pat": pat,
                    "chan": chan,
                    "time": (t_start + t_end) / 2,
                }
                | {
                    lag: val
                    for lag, val in zip(
                        patTAU.acf_data["lags"], patTAU.acf_data["acf_avg"][0]
                    )
                },
                index=[0],
            ))

        df_timescales_sw_chan = pd.concat(df_timescales_sw_chan, ignore_index=True)
        df_acf_sw_chan = pd.concat(df_acf_sw_chan, ignore_index=True)

        return df_timescales_sw_chan, df_acf_sw_chan

    def _compute_timescales_pat(self, pat):

        # Keep info about every channel
        df_timescales_sw_pat = []
        df_acf_sw_pat = []

        # Create epochs from SW events
        raw_pat = self.raws[pat]
        df_events = pd.read_csv(
            self.sw_path.joinpath(pat, f"sw_events_{self.stage}.csv"), index_col=0
        )
        epochs_pat = utils_sw._epoch_sws(
            df_events,
            raw_pat.get_data(),
            raw_pat.ch_names,
            raw_pat.info["sfreq"],
            center="NegPeak",
            t_around=self.epo_sws,
        )

        # Create folders & files
        utils_acf.create_pat_folder(pat, self.results_path, raw_pat.ch_names)

        # Prepare data analyses
        patACF = ACF(self.config_path, pat)
        patTAU = TAU(self.config_path, pat)
        # Loop through every channel
        for chan in epochs_pat.keys():
            epo_chan = epochs_pat[chan]

            df_timescales_sw_chan, df_acf_sw_chan = self._compute_timescales_chan(
                epo_chan, pat, chan, raw_pat.info["sfreq"], patACF, patTAU
            )
            df_timescales_sw_pat.append(df_timescales_sw_chan)
            df_acf_sw_pat.append(df_acf_sw_chan)

        # Concatenate results
        df_timescales_sw_pat = pd.concat(df_timescales_sw_pat, ignore_index=True)
        df_acf_sw_pat = pd.concat(df_acf_sw_pat, ignore_index=True)

        return df_timescales_sw_pat, df_acf_sw_pat

    def compute_timescales_sliding(
        self,
        epo_sws: float,
        epo_dur: float,
        epo_overlap: float,
        nlags: int,
        tau_mode: str,
        fit_func: str,
        fit_range: list,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            nlags (int): _description_
            tau_mode (str): _description_
            fit_func (str): _description_
            fit_range (list): _description_

        Returns:
            pd.DataFrame: _description_
        """

        self.epo_sws = epo_sws
        self.epo_dur = epo_dur
        self.epo_overlap = epo_overlap
        self.nlags = nlags
        self.tau_mode = tau_mode
        self.fit_func = fit_func
        self.fit_range = fit_range

        # Dataframes for all results
        df_timescales_sw = []
        df_acf_sw = []

        for pat in self.raws.keys():
            print("Patient: ", pat)

            df_info_pat = self.df_info[self.df_info["pat"] == pat]

            # Check Raw is available
            if self.raws[pat] is None:
                continue
            timescales_sw_pat, df_acf_sw_pat = self._compute_timescales_pat(pat)
            col_res = timescales_sw_pat["time"].unique().tolist()
            df_timescales_sw_pat = create_res_df(
                df_info_pat, self.raws[pat].ch_names, self.stage, columns_res=col_res
            )
            df_timescales_sw_pat.loc[:, col_res] = timescales_sw_pat.pivot(
                columns="time", index="chan", values="tau"
            ).loc[df_timescales_sw_pat["chan"], col_res].to_numpy()

            df_timescales_sw.append(df_timescales_sw_pat)
            df_acf_sw.append(df_acf_sw_pat)

        # Concatenate results
        df_timescales_sw = pd.concat(df_timescales_sw, ignore_index=True)
        df_acf_sw = pd.concat(df_acf_sw, ignore_index=True)

        return df_timescales_sw, df_acf_sw
