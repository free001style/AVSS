import torch
from torch import nn
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class SISNR(nn.Module):
    """
    SI-SNR loss class.
    """

    def __init__(self):
        super().__init__()
        self.loss = scale_invariant_signal_noise_ratio

    def forward(self, source, predict, **batch):
        """
        Loss function calculation logic.
        Args:
            source (Tensor): (B, n_spk, T) tensor of ground truth speech.
            predict (Tensor): (B, n_spk, T) tensor of predicted speech.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        loss = self.loss(predict, source).mean()
        return {"loss": loss}
