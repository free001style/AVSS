import torch
import torch.nn as nn
from torch.nn import functional as F

from src.model.layers.attention import TFAttention
from src.model.layers.conv import Conv
from src.model.layers.dual_path import DualPath
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class RTFSBlock(nn.Module):
    """
    RTFS Block:
        1) Compression phase
        2) Processing frequency with DualPath
        3) Processing time with DualPath
        4) Reconstruction phase.
    """

    def __init__(
        self,
        channel_dim,
        hidden_dim,
        freqs,
        q=2,
    ):
        """
        Args:
            channel_dim (int): audio channel dimension (C_a in paper).
            hidden_dim (int): channel dimension in RTFS block (D in paper).
            freqs (int): number of frequencies (needs for cfLN).
            q (int): number of spacial dim decreasing in compression phase.
        """
        super(RTFSBlock, self).__init__()
        self.dualpath_f = DualPath()
        self.dualpath_t = DualPath()
        self.tf_attention = TFAttention(freqs=freqs // (2 ** (q - 1)))
        self.proj = Conv(
            channel_dim,
            hidden_dim,
            kernel_size=1,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.back_proj = Conv(hidden_dim, channel_dim, kernel_size=1)
        self.compress = Compressor(q, hidden_dim)
        self.decompress = Decompressor(q, hidden_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C_a, T_a, F) tensor of fused features.
        Returns:
            predict: (B, C_a, T_a, F).
        """
        residual = x
        x = self.proj(x)
        downsample_list, x = self.compress(x)
        x = self.dualpath_f(x)
        x = x.transpose(2, 3)
        x = self.dualpath_t(x)
        x = x.transpose(2, 3)
        x = self.tf_attention(x)
        x = self.decompress(x, downsample_list)
        x = self.back_proj(x) + residual
        return x


class Compressor(nn.Module):
    def __init__(self, q, channel_dim, is_2d=True):
        super(Compressor, self).__init__()
        self.q = q
        self.is_2d = is_2d
        self.downsample = nn.ModuleList(
            [
                Conv(
                    channel_dim,
                    channel_dim,
                    kernel_size=4,
                    stride=2 if i > 0 else 1,
                    padding="same" if i == 0 else None,
                    groups=channel_dim,
                    is_2d=is_2d,
                    activation=nn.PReLU,
                    normalization=gLN,
                )
                for i in range(q)
            ]
        )
        self.pooling = F.adaptive_avg_pool2d if is_2d else F.adaptive_avg_pool1d

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, *).
        Returns:
            downsample_list (List[Tensor]): list of downsampled tensors.
            predict (Tensor): (B, D, *).
        """
        downsample_list = [self.downsample[0](x)]
        for i in range(self.q - 1):
            downsample_list.append(self.downsample[i + 1](downsample_list[-1]))
        x = downsample_list[-1]
        for i in range(len(downsample_list) - 1):
            x += self.pooling(downsample_list[i], (x.shape[2:]))

        return downsample_list, x


class Decompressor(nn.Module):
    def __init__(self, q, channel_dim, is_2d=True):
        super(Decompressor, self).__init__()
        self.is_2d = is_2d
        self.q = q
        self.first_phase = nn.ModuleList(
            [Interpolation(channel_dim, is_2d) for i in range(q)]
        )
        self.second_phase = nn.ModuleList(
            [Interpolation(channel_dim, is_2d) for i in range(q - 1)]
        )

    def forward(self, x, downsample_list):
        """
        Args:
            x (Tensor): (B, D, *).
            downsample_list (List[Tensor]): list of downsampled tensors.
        Returns:
            predict (Tensor): (B, C, *).
        """
        i_list = [
            self.first_phase[i](downsample_list[i], x)
            for i in range(len(downsample_list))
        ]
        x = self.second_phase[-1](i_list[-2], i_list[-1]) + downsample_list[-2]
        for i in range(len(downsample_list) - 3, -1, -1):
            x = self.second_phase[i](i_list[i], x) + downsample_list[i]
        return x


class Interpolation(nn.Module):
    def __init__(self, channel_dim, is_2d=True):
        super(Interpolation, self).__init__()
        self.is_2d = is_2d
        self.w1 = Conv(
            channel_dim,
            channel_dim,
            4,
            padding="same",
            groups=channel_dim,
            is_2d=is_2d,
            normalization=gLN,
            activation=nn.Sigmoid,
        )
        self.w2 = Conv(
            channel_dim,
            channel_dim,
            4,
            padding="same",
            groups=channel_dim,
            is_2d=is_2d,
            normalization=gLN,
        )
        self.w3 = Conv(
            channel_dim,
            channel_dim,
            4,
            padding="same",
            groups=channel_dim,
            is_2d=is_2d,
            normalization=gLN,
        )

    def forward(self, m, n):
        """
        Args:
            m (Tensor): (B, D, *).
            n (Tensor): (B, D, *).
        Returns:
            predict (Tensor): (B, D, *).
        """
        m = self.w2(m)
        n1 = F.interpolate(self.w1(n), size=m.shape[2:], mode="nearest")
        n2 = F.interpolate(self.w3(n), size=m.shape[2:], mode="nearest")
        return n1 * m + n2
