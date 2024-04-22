from pathlib import Path
import itertools
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
from .utils import create_RawMNE, create_epo


class Load:

    def __init__(
        self,
        mat_path: Path,
    ):

        self.mat_path = mat_path

        self.raw_data = None
        self.df_info = None
        self.epochs = {}

    def _load_raw_data(self):

        self.raw_data = loadmat(self.mat_path)

    def get_info(self):

        if self.raw_data is None:
            self._load_raw_data()

        ch_names = self.raw_data["ChannelName"].squeeze()
        ch_types = self.raw_data["ChannelType"].squeeze()
        ch_regs = self.raw_data["ChannelRegion"].squeeze()
        pat_ids = self.raw_data["Patient"].squeeze()
        genders = self.raw_data["Gender"].squeeze()
        ages = self.raw_data["AgeAtTimeOfStudy"].squeeze()
        ch_pos = self.raw_data["ChannelPosition"].squeeze()
        regions_map = {
            i + 1: r[0][0] for i, r in enumerate(self.raw_data["RegionName"])
        }

        # Put info together in dataframe
        self.df_info = pd.DataFrame(
            {
                "pat": pat_ids,
                "age": list(
                    itertools.chain.from_iterable(
                        [
                            [age] * (pat_ids == pat_id).sum()
                            for pat_id, age in zip(set(pat_ids), ages)
                        ]
                    )
                ),
                "gender": list(
                    itertools.chain.from_iterable(
                        [
                            [gender] * (pat_ids == pat_id).sum()
                            for pat_id, gender in zip(set(pat_ids), genders)
                        ]
                    )
                ),
                "chan": [ch[0] for ch in ch_names],
                "type": [t[0] for t in ch_types],
                "region": ch_regs,
                "mni_x": ch_pos[:, 0],
                "mni_y": ch_pos[:, 1],
                "mni_z": ch_pos[:, 2],
            }
        )
        self.df_info["region"] = self.df_info["region"].apply(lambda x: regions_map[x])

    def load_epo_stage(
        self,
        stage,
        epo_dur=1.0,
        epo_overlap=0.5,
        filt=False,
        filt_freqs=None,
    ):

        # Load info if not present
        if self.df_info is None:
            self.get_info()
        sfreq = self.raw_data["SamplingFrequency"][0][0]
        assert stage in [
            "W",
            "N2",
            "N3",
            "R",
        ], "Invalid stage! Stage can be W, N2, N3, or R."
        data_stage = self.raw_data["Data_" + stage].T

        # Loop through patients to extract epochs
        pats = self.df_info["pat"].unique().tolist()
        for pat in tqdm(pats, total=len(pats)):
            pat_code = str(pat).zfill(3)
            df_info_pat = self.df_info[self.df_info["pat"] == pat]
            chans_pat = df_info_pat["chan"].to_list()
            # Raw MNE object
            data_stage_pat = data_stage[df_info_pat.index]
            raw_stage_pat = create_RawMNE(
                data_stage_pat, chans_pat, sfreq
            )

            # Get epochs
            if raw_stage_pat is None:
                epo_stage_pat = None
            else:
                epo_stage_pat = create_epo(
                    raw_stage_pat,
                    epo_dur=epo_dur,
                    epo_overlap=epo_overlap,
                    freq_band=filt,
                    band_freqs=filt_freqs,
                )
            # Append to dictionary
            self.epochs[pat_code] = epo_stage_pat
