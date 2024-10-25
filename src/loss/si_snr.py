import torch
from torch import nn
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from .loss_utils import si_snr


class SI_SNR(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = si_snr

    def forward(self, source, predict, **batch):
        """
        Loss function calculation logic.
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        
        return {"loss": self.loss(predict, source)}
