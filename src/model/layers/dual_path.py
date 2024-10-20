from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from sru import SRU

from src.model.layers import cLN


class DualPath(nn.Module):
    def __init__(
        self, in_channels=64, kernel_size=8, stride=1, hidden_dim=32, num_rnn=4
    ):
        super(DualPath, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.unfold = nn.Unfold((kernel_size, 1), stride=(stride, 1))
        self.normalization = cLN(kernel_size * in_channels)
        self.sru = SRU(
            kernel_size * in_channels, hidden_dim, num_rnn, bidirectional=True
        )
        self.tconv = nn.ConvTranspose1d(
            2 * hidden_dim, in_channels, kernel_size, stride
        )

    def forward(self, x):
        residual = x
        b, c, t, f = x.shape
        f_pad = (
            ceil((f - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        )
        x = F.pad(x, (0, f_pad - f), "constant", 0)  # b x D x t x f_pad
        x = x.permute(0, 2, 1, 3).contiguous().view(b * t, c, f_pad)
        x = self.unfold(x[..., None])  # b * t x 8D x ...
        x = self.normalization(x)
        # since sru don't have batch_first, we have to transpose among other bt and ...
        x = x.permute(2, 0, 1)  # ... x b * t x 8D
        x, _ = self.sru(x)  # ... x b * t x 2 * h
        x = x.permute(1, 2, 0)  # b * t x 2 * h x ...
        x = self.tconv(x)  # b * t x D x f
        x = x.view(b, t, c, f).transpose(1, 2)
        return x + residual
