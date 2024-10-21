import torch
import torch.nn as nn

from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class AudioEncoder(nn.Module):
    def __init__(self, channel_dim=256, win_length=255, hop_length=128):
        super(AudioEncoder, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.conv = Conv(2, channel_dim, 3, normalization=gLN, activation=nn.PReLU)
        self.window = torch.hann_window(self.win_length)

    def forward(self, audio):
        alpha = torch.view_as_real(
            torch.stft(
                audio,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                window=self.window.to(audio.device),
                return_complex=True,
            )
        )  # (b, freq, time, 2)
        alpha = self.conv(alpha.permute(0, 3, 2, 1))  # (b, c, time, freq)
        return alpha


class AudioDecoder(nn.Module):
    def __init__(self, channel_dim=256, win_length=255, hop_length=128):
        super(AudioDecoder, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.conv = nn.ConvTranspose2d(
            channel_dim, 2, kernel_size=(3, 3), padding=(1, 1), bias=False
        )
        nn.init.xavier_uniform_(self.conv.weight)
        self.window = torch.hann_window(self.win_length)

    def forward(self, x, length):
        b, n_speak, c, t, f = x.shape
        x = self.conv(x.view(b * n_speak, c, t, f))  # (b, c, time, freq)
        x = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2)
        audio = torch.istft(
            x,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # (b, length)
        return audio.view(b, n_speak, length)
