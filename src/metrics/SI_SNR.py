import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio



class SI_SNR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Metric calculation logic.

        Args:
            source (Tensor): ground-truth waveforms.
            prediction (Tensor): predicted waveforms.
        Returns:
            si-snr (float): calculated si-snr.
        """
        if source.size() != predict.size() or source.ndim != 3:
            raise TypeError(f"Inputs must be of shape [batch, n_src, time], got {source.size()} and {predict.size()} instead")
        
        return scale_invariant_signal_noise_ratio(predict, source)
        
