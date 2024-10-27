import torch

from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies sdri metric function

        """
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, mix, **kwargs):
        """
        Metric calculation logic.

        Args:
            source (Tensor): ground-truth waveforms.
            predict (Tensor): predicted waveforms.
            mix (Tensor): mixed waveform.

        Returns:
            sdri (float): calculated sdri.
        """
        if source.size() != predict.size() or source.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {source.size()} and {predict.size()} instead"
            )

        return torch.mean(sdr(predict, source) - sdr(mix.repeat(2, 1, 1).transpose(1, 0), source))
