import torch
import torch.nn as nn


class ChannelFrequencyLayerNorm(nn.Module):
    def __init__(self, dim):
        super(ChannelFrequencyLayerNorm, self).__init__()
        self.normalization = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b * t, c, f)
        x = self.normalization(x)
        x = x.view(b, t, c, f).transpose(1, 2)
        return x


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim):
        super(ChannelLayerNorm, self).__init__()
        self.normalization = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b * f, c)
        x = self.normalization(x)
        x = x.view(b, f, c).transpose(1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim):
        super(GlobalLayerNorm, self).__init__()
        self.normalization = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.normalization(x)
