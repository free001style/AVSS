import torch
import torch.nn as nn

from src.model.layers.normalizations import ChannelFrequencyLayerNorm as cfLN


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        freqs=None,
        normalization=None,
        activation=None,
        is_2d=True,
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d if is_2d else nn.Conv1d
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = self.conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain("leaky_relu"))
        self.activation = activation() if activation is not None else nn.Identity()
        if normalization is not None:
            self.normalization = (
                normalization((out_channels, freqs))
                if normalization == cfLN
                else normalization(out_channels)
            )
        else:
            self.normalization = nn.Identity()

    def forward(self, x):
        return self.activation(self.normalization(self.conv(x)))
