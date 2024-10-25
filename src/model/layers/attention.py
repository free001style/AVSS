from math import log, sqrt

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.model.layers.conv import Conv
from src.model.layers.normalizations import ChannelFrequencyLayerNorm as cfLN
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class TFAttention(nn.Module):
    def __init__(self, freqs, channel_dim=64, hidden_dim=8, n_heads=4):
        super(TFAttention, self).__init__()
        self.channel_dim = channel_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.q_proj = nn.ModuleList(
            [
                Conv(
                    channel_dim,
                    hidden_dim,
                    1,
                    activation=nn.PReLU,
                    normalization=cfLN,
                    freqs=freqs,
                )
                for i in range(n_heads)
            ]
        )
        self.k_proj = nn.ModuleList(
            [
                Conv(
                    channel_dim,
                    hidden_dim,
                    1,
                    activation=nn.PReLU,
                    normalization=cfLN,
                    freqs=freqs,
                )
                for i in range(n_heads)
            ]
        )
        self.v_proj = nn.ModuleList(
            [
                Conv(
                    channel_dim,
                    channel_dim // n_heads,
                    1,
                    activation=nn.PReLU,
                    normalization=cfLN,
                    freqs=freqs,
                )
                for i in range(n_heads)
            ]
        )
        self.proj = Conv(
            channel_dim,
            channel_dim,
            1,
            activation=nn.PReLU,
            normalization=cfLN,
            freqs=freqs,
        )

    def forward(self, x):
        b, c, t, f = x.shape
        residual = x
        all_q = [q(x) for q in self.q_proj]
        all_k = [k(x) for k in self.k_proj]
        all_v = [v(x) for v in self.v_proj]

        Q = torch.cat(all_q, dim=0)  # h_head * b x E x t x f
        K = torch.cat(all_k, dim=0)  # h_head * b x E x t x f
        V = torch.cat(all_v, dim=0)  # h_head * b x D / n_head x t x f

        Q = (
            Q.transpose(1, 2)
            .contiguous()
            .view(self.n_heads * b, t, self.hidden_dim * f)
        )
        K = (
            K.transpose(1, 2)
            .contiguous()
            .view(self.n_heads * b, t, self.hidden_dim * f)
        )
        V = (
            V.transpose(1, 2)
            .contiguous()
            .view(self.n_heads * b, t, self.channel_dim // self.n_heads * f)
        )

        QK = torch.matmul(Q, K.transpose(1, 2)) / sqrt(
            f * self.hidden_dim
        )  # h_head * b x t x t
        QK_softmax = F.softmax(QK, dim=2)
        A = torch.matmul(QK_softmax, V)  # h_head * b x t x D / n_head * f

        A = A.view(self.n_heads * b, t, self.channel_dim // self.n_heads, f).transpose(
            1, 2
        )  # h_head * b x D / n_head x t x f
        A = (
            A.view(self.n_heads, b, self.channel_dim // self.n_heads, t, f)
            .transpose(0, 1)
            .contiguous()
        )
        A = A.view(b, self.channel_dim, t, f)
        out = self.proj(A)
        return out + residual


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = torch.zeros((max_len, embed_dim))
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-log(max_len) / embed_dim)
        ).unsqueeze(0)
        arguments = positions * freq
        self.pos_encoding[:, 0::2] = torch.sin(arguments)
        self.pos_encoding[:, 1::2] = torch.cos(arguments)
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[None, : x.shape[1]])


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, numb_head=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.normalization = nn.RMSNorm(embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.mhsa = nn.MultiheadAttention(
            embed_dim, numb_head, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # embed_dim is features and t is seq
        x = self.pos_encoding(x)
        residual = x
        x = self.normalization(x)
        x, _ = self.mhsa(x, x, x)
        self.dropout(x) + residual
        x = x.transpose(1, 2)
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.conv1 = Conv(
            embed_dim,
            embed_dim * 2,
            1,
            is_2d=False,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.conv2 = Conv(
            embed_dim * 2,
            embed_dim * 2,
            5,
            groups=embed_dim * 2,
            is_2d=False,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.conv3 = Conv(
            embed_dim * 2,
            embed_dim,
            1,
            is_2d=False,
            activation=nn.PReLU,
            normalization=gLN,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dropout1(self.conv1(x))
        x = self.dropout2(self.conv2(x))
        x = self.dropout3(self.conv3(x))
        return x + residual


class VideoAttention(nn.Module):
    def __init__(self, embed_dim):
        super(VideoAttention, self).__init__()
        self.MHSA = MultiHeadAttention(embed_dim)
        self.FFN = FFN(embed_dim)

    def forward(self, x):
        x = self.MHSA(x)
        x = self.FFN(x)
        return x
