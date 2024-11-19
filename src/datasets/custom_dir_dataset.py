import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        self._data_dir = ROOT_PATH / data_dir
        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        index = []
        audio_dir = self._data_dir / "audio"
        mouths_dir = self._data_dir / "mouths"
        for mix in os.listdir(str(audio_dir / "mix")):
            if mix.endswith((".wav", ".mp3", ".flac")):
                mix_path = str(audio_dir / "mix" / mix)
                if (audio_dir / "s1").exists():
                    s1_path = str(audio_dir / "s1" / mix)
                    s2_path = str(audio_dir / "s2" / mix)
                    self.separate_only = False
                else:
                    s1_path = s2_path = None
                    self.separate_only = True
                if mouths_dir.exists():
                    mouths1, mouths2 = (
                        mix[:-5].split("_")
                        if mix.endswith(".flac")
                        else mix[:-4].split("_")
                    )
                    mouths1_path = str(mouths_dir / f"{mouths1}.npz")
                    mouths2_path = str(mouths_dir / f"{mouths2}.npz")
                else:
                    mouths1_path = mouths2_path = None
                index.append(
                    {
                        "mix": mix_path,
                        "label1": s1_path,
                        "label2": s2_path,
                        "mouths1": mouths1_path,
                        "mouths2": mouths2_path,
                        "name": mix,
                    }
                )
        return index
