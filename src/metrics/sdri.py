import torch
from torchmetrics.functional.audio import signal_distortion_ratio as sdr

from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies SDRi metric function.
        """
        self.metric = sdr
        self.use_pit = kwargs["use_pit"]
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, mix, **batch):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
            mix (Tensor): (B, T) mixed speech.
        Returns:
            metric (Tensor): calculated SDRi.
        """
        if self.use_pit and self.metric.device != source.device:
            self.metric = self.metric.to(source.device)
        return torch.mean(
            self.metric(predict, source)
            - self.metric(mix.unsqueeze(1).expand(-1, source.shape[1], -1), source)
        )
