import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from src.metrics.base_metric import BaseMetric


class SISNRi(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies SI-SNRi metric function.
        """
        self.metric = si_snr
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, mix, **batch):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
            mix (Tensor): (B, T) mixed speech.
        Returns:
            SI-SNRi (Tensor): calculated SI-SNRi.
        """
        return torch.mean(
            self.metric(predict, source)
            - self.metric(mix.unsqueeze(1).expand(-1, source.shape[1], -1), source)
        )
