import torch
import torch.nn as nn

from src.model.layers.attention import VideoAttention
from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN
from src.model.layers.rtfs_block import Compressor, Decompressor


class VPBlock(nn.Module):
    """
    Block for video processing. Lightweight version of RTFS block but TF Attention has been replaced with
    Multi Head Self-Attention.
    """

    def __init__(self, q=4, embed_dim=512, hidden_dim=64):
        """
        Args:
            q (int): number of spacial dim decreasing in compression phase (q in paper).
            embed_dim (int): video embedding dimension (C_v in paper).
            hidden_dim (int): channel dimension in RTFS block (D in paper).
        """
        super(VPBlock, self).__init__()
        self.proj = Conv(
            embed_dim,
            hidden_dim,
            kernel_size=1,
            is_2d=False,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.compress = Compressor(q, hidden_dim, False)
        self.attention = VideoAttention(hidden_dim)
        self.decompress = Decompressor(q, hidden_dim, False)
        self.back_proj = Conv(hidden_dim, embed_dim, kernel_size=1, is_2d=False)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, T) video embeddings.
        Returns:
            predict (Tensor): (B, C, T).
        """
        residual = x
        x = self.proj(x)  # b x d x t
        downsample_list, x = self.compress(x)
        x = self.attention(x)
        x = self.decompress(x, downsample_list)
        x = self.back_proj(x) + residual
        return x
