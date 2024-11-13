import torch
import torch.nn as nn

from src.model.layers.conv import Conv
from src.model.layers.normalizations import GlobalLayerNorm as gLN


class AudioEncoder(nn.Module):
    """
    Module for audio encoding.
    """

    def __init__(self, channel_dim=256, n_fft=255, hop_length=128):
        """
        Args:
            channel_dim (int): audio channel dimension (C_a in paper).
            n_fft (int): n_fft for stft.
            hop_length (int): hop_length for stft.
        """
        super(AudioEncoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv = Conv(2, channel_dim, 3)
        self.window = torch.hann_window(n_fft)

    def forward(self, audio):
        """
        Args:
             audio (Tensor): (B, L) tensor of mix audio.
        Returns:
            predict (Tensor): (B, C, Time, Freq) tensor of encoded audio.
        """
        alpha = torch.view_as_real(
            torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.to(audio.device),
                return_complex=True,
            )
        )  # (b, freq, time, 2)
        return self.conv(alpha.permute(0, 3, 2, 1))


class AudioDecoder(nn.Module):
    """
    Module for audio decoding.
    """

    def __init__(self, channel_dim=256, n_fft=255, hop_length=128):
        super(AudioDecoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv = nn.ConvTranspose2d(
            channel_dim, 4, kernel_size=3, padding=1, bias=False
        )
        nn.init.xavier_uniform_(self.conv.weight)
        self.window = torch.hann_window(n_fft)

    def forward(self, x, length):
        """
        Args:
            x (Tensor): (B, C, Time, Freq) tensor of masked features.
            length (Tensor[int]): (B, ) length of predicted audio.
        Returns:
            audio (Tensor): (B, L) tensor of predicted audio.
        """
        x = self.conv(x)  # (b, c, time, freq)
        x1 = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2)
        audio1 = torch.istft(
            x1,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )
        x2 = torch.complex(x[:, 2], x[:, 3]).transpose(1, 2)
        audio2 = torch.istft(
            x2,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )
        return audio1, audio2
