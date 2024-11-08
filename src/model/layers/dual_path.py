from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers.normalizations import ChannelLayerNorm as cLN


class DualPath(nn.Module):
    """
    Implementation of the Dual Path from https://arxiv.org/abs/2209.03952:
        1) Pad dimension for processing
        2) Unfold
        3) Process dimension with rnn
        4) Upsample features for input shape.
    """

    def __init__(
        self, in_channels=64, kernel_size=8, stride=1, hidden_dim=32, num_rnn=4
    ):
        """
        Args:
            in_channels (int): D in paper.
            kernel_size (int): kernel size of unfold.
            stride (int): stride of unfold.
            hidden_dim (int): hidden dimension of rnn layer (H in paper).
            num_rnn (int): number of rnn layers.
        """
        super(DualPath, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.unfold = nn.Unfold((kernel_size, 1), stride=(stride, 1))
        self.normalization = cLN(kernel_size * in_channels)
        self.sru = nn.LSTM(
            kernel_size * in_channels,
            hidden_dim,
            num_rnn,
            bidirectional=True,
        )  # TODO dropout?
        self.tconv = nn.ConvTranspose1d(
            2 * hidden_dim, in_channels, kernel_size, stride
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, D, T, F)
        Returns:
            predict (Tensor): (B, D, T, F)
        """
        residual = x
        b, c, t, f = x.shape
        f_pad = (
            ceil((f - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        )
        x = F.pad(x, (0, f_pad - f), "constant", 0)  # b x D x t x f_pad
        x = x.transpose(1, 2).contiguous().view(b * t, c, f_pad)
        x = self.unfold(x[..., None])  # b * t x 8D x ...
        x = self.normalization(x)
        # since sru don't have batch_first, we have to transpose among other bt and ...
        x = x.permute(2, 0, 1).contiguous()  # ... x b * t x 8D
        x = self.sru(x)[0]  # ... x b * t x 2 * h
        x = x.permute(1, 2, 0)  # b * t x 2 * h x ...
        x = self.tconv(x)  # b * t x D x f
        x = x.view(b, t, c, f).transpose(1, 2)
        return x[..., :f] + residual
