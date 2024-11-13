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

    def __init__(self, channel_dim=256, n_fft=255, hop_length=128, use_video=True):
        super(AudioDecoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_video = use_video
        self.out_channels = 2 if use_video else 4
        self.conv = nn.ConvTranspose2d(
            channel_dim, self.out_channels, kernel_size=3, padding=1, bias=False
        )
        nn.init.xavier_uniform_(self.conv.weight)
        self.window = torch.hann_window(n_fft)

    def _get_audio(self, x, length):
        x = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2)
        audio = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )
        return audio

    def forward(self, x, length):
        """
        Args:
            x (Tensor): (B, C, Time, Freq) tensor of masked features.
            length (Tensor[int]): (B, ) length of predicted audio.
        Returns:
            audio (Tensor): (B, L) tensor of predicted audio.
        """
        x = self.conv(x)  # (b, c, time, freq)
        if not self.use_video:
            audio1 = self._get_audio(x[:, :2], length)
            audio2 = self._get_audio(x[:, 2:], length)
            return torch.cat((audio1, audio2))
        else:
            audio = self._get_audio(x, length)
            return audio
