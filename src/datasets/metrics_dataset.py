import os

import torch
import torchaudio
from torch.utils.data import Dataset


class MetricsDataset(Dataset):
    def __init__(self, predict_dir, gt_dir):
        """
        Args:
            predict_dir (Path):
            gt_dir (Path):
        """
        super().__init__()
        self._mix, self._predict1, self._predict2, self._gt1, self._gt2 = (
            [],
            [],
            [],
            [],
            [],
        )
        for mix in os.listdir(predict_dir / "mix"):
            if mix.endswith((".wav", ".mp3", ".flac")):
                self._mix.append(predict_dir / "mix" / mix)
                self._predict1.append(predict_dir / "s1" / mix)
                self._predict2.append(predict_dir / "s2" / mix)
                self._gt1.append(gt_dir / "s1" / mix)
                self._gt2.append(gt_dir / "s2" / mix)

    def __len__(self):
        return len(self._predict1)

    def __getitem__(self, ind):
        mix = self.load_audio(self._mix[ind]).squeeze()
        predict1 = self.load_audio(self._predict1[ind]).squeeze()
        predict2 = self.load_audio(self._predict2[ind]).squeeze()
        gt1 = self.load_audio(self._gt1[ind]).squeeze()
        gt2 = self.load_audio(self._gt2[ind]).squeeze()
        return {
            "mix": mix,
            "predict": torch.stack([predict1, predict2]),
            "source": torch.stack([gt1, gt2]),
        }

    @staticmethod
    def load_audio(path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = 16000
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
