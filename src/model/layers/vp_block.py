import torch
import torch.nn as nn

from src.model.layers.attention import VideoAttention
from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN
from src.model.layers.rtfs_block import Compressor, Decompressor


class VPBlock(nn.Module):
    def __init__(self, q=4, embed_dim=512, hidden_size=64, is_2d=False):
        super(VPBlock, self).__init__()
        self.proj = Conv(
            embed_dim,
            hidden_size,
            kernel_size=1,
            is_2d=is_2d,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.compress = Compressor(q, hidden_size, is_2d)
        self.attention = VideoAttention(hidden_size)
        self.decompress = Decompressor(q, hidden_size, is_2d)
        self.back_proj = Conv(hidden_size, embed_dim, kernel_size=1, is_2d=is_2d)

    def forward(self, x):
        residual = x  # b x c x t
        x = self.proj(x)  # b x d x t
        downsample_list, x = self.compress(x)
        x = self.attention(x)
        x = self.decompress(x, downsample_list)
        x = self.back_proj(x) + residual
        return x
