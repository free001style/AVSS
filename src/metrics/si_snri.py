import torch

from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from src.metrics.base_metric import BaseMetric


class SISNRi(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies si-snri metric function

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
            si-snri (float): calculated si-snri.
        """
        if source.size() != predict.size() or source.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {source.size()} and {predict.size()} instead"
            )

        return torch.mean(si_snr(predict, source) - si_snr(mix.repeat(2, 1, 1).transpose(1, 0), source))
