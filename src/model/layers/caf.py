import torch
import torch.nn.functional as F
from torch import nn

from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class CAFBlock(nn.Module):
    def __init__(self, cin_a, cin_v, h):
        super(CAFBlock, self).__init__()
        self.h = h
        self.cin_a = cin_a
        self.conv_gate = Conv(
            cin_a,
            cin_a,
            1,
            groups=cin_a,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU,
        )
        self.conv_val = Conv(
            cin_a, cin_a, 1, groups=cin_a, normalization=nn.BatchNorm2d
        )

        self.conv_attn = Conv(
            cin_v, cin_a * self.h, 1, groups=cin_a, is_2d=False, normalization=gLN
        )
        self.conv_key = Conv(
            cin_v, cin_a, 1, groups=cin_a, is_2d=False, normalization=gLN
        )

    def forward(self, a1, v1):
        # a1 = b x Ca x Ta x F
        # v1 = b x Cv x Tv
        b, _, time, _ = a1.shape
        v_key = F.interpolate(self.conv_key(v1), size=time, mode="nearest")
        v1 = self.conv_attn(v1)  # b x cin_a*h x Tv
        vm = torch.mean(v1.view(b, self.cin_a, self.h, -1), dim=2, keepdim=False).view(
            b, self.cin_a, -1
        )
        v_attn = F.interpolate(F.softmax(vm, -1), size=time, mode="nearest")

        a_val = self.conv_val(a1)
        a_gate = self.conv_gate(a1)

        f1 = v_attn[..., None] * a_val
        f2 = v_key[..., None] * a_gate
        return f1 + f2  # Ca x Ta x F
