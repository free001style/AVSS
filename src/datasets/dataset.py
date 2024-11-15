import json
import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from pathlib import Path


class Dataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "dla_dataset"
        elif isinstance(data_dir, str):
            data_dir = Path(data_dir).absolute()
        self._data_dir = data_dir
        save_dir = kwargs.get('save_dir', data_dir)
        if isinstance(save_dir, str):
            save_dir = Path(save_dir).absolute()
        self._save_dir = save_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._save_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
                self.separate_only = (
                    True if index[0]["label1"] == index[0]["label2"] else False
                )
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        audio_dir = self._data_dir / "audio" / part
        mouths_dir = self._data_dir / "mouths"
        for mix in os.listdir(str(audio_dir / "mix")):
            if mix.endswith(".wav"):
                mix_path = str(audio_dir / "mix" / mix)
                if (audio_dir / "s1").exists():
                    s1_path = str(audio_dir / "s1" / mix)
                    s2_path = str(audio_dir / "s2" / mix)
                    self.separate_only = False
                else:
                    s1_path = s2_path = str(mix)
                    self.separate_only = True
                mouths1, mouths2 = mix[:-4].split("_")
                mouths1_path = str(mouths_dir / f"{mouths1}.npz")
                mouths2_path = str(mouths_dir / f"{mouths2}.npz")
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
