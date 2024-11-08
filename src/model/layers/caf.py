import torch
import torch.nn.functional as F
from torch import nn

from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class CAFBlock(nn.Module):
    """
    Cross-Dimensional Attention Fusion block
    """

    def __init__(self, channel_dim_a, channel_dim_v, h):
        """
        Args:
            channel_dim_a (int): audio channel dimension (C_a in paper).
            channel_dim_v (int): video embedding dimension (C_v in paper).
            h (int): number of heads.
        """
        super(CAFBlock, self).__init__()
        self.h = h
        self.channel_dim_a = channel_dim_a
        self.conv_gate = Conv(
            channel_dim_a,
            channel_dim_a,
            1,
            groups=channel_dim_a,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
        )
        self.conv_val = Conv(
            channel_dim_a,
            channel_dim_a,
            1,
            groups=channel_dim_a,
            normalization=nn.BatchNorm2d,
        )

        self.conv_attn = Conv(
            channel_dim_v,
            channel_dim_a * self.h,
            1,
            groups=channel_dim_a,
            is_2d=False,
            normalization=gLN,
        )
        self.conv_key = Conv(
            channel_dim_v,
            channel_dim_a,
            1,
            groups=channel_dim_a,
            is_2d=False,
            normalization=gLN,
        )

    def forward(self, a1, v1):
        """
        Args:
            a1 (Tensor): (B, C_a, T_a, F) audio embedding after AP block.
            v1 (Tensor): (B, C_v, T_v) video embedding after VP block.
        Returns:
            fused (Tensor): (B, C_a, T_a, F).
        """
        b, c, t, f = a1.shape
        a_val = self.conv_val(a1)
        a_gate = self.conv_gate(a1)
        v_key = F.interpolate(self.conv_key(v1), size=t, mode="nearest")
        v1 = self.conv_attn(v1)  # b x cin_a*h x Tv
        vm = torch.mean(
            v1.view(b, self.channel_dim_a, self.h, -1),
            dim=2,
        ).view(b, self.channel_dim_a, -1)
        v_attn = F.interpolate(F.softmax(vm, -1), size=t, mode="nearest")

        f1 = v_attn[..., None] * a_val
        f2 = v_key[..., None] * a_gate
        return f1 + f2
