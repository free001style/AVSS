import torch
from torch import nn
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class SISNR(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = scale_invariant_signal_noise_ratio

    def forward(self, source, predict, **batch):
        """
        Loss function calculation logic.
        Args:
            predict (Tensor): model output predictions.
            source (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(predict, source).mean()}
