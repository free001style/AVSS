import torch

from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from src.metrics.base_metric import BaseMetric


class SISNR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies si-snr metric function
        """
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Metric calculation logic.

        Args:
            source (Tensor): ground-truth waveforms.
            predict (Tensor): predicted waveforms.
        Returns:
            si-snr (float): calculated si-snr.
        """
        if source.size() != predict.size() or source.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {source.size()} and {predict.size()} instead"
            )

        return si_snr(predict, source).mean()
