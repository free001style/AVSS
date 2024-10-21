import torch
from torch import nn

from src.model.layers.conv import Conv


class S3(nn.Module):
    def __init__(self, channel_dim, n_speakers=2):
        super(S3, self).__init__()
        self.n_speakers = n_speakers
        self.prelu = nn.PReLU()
        self.conv = Conv(channel_dim, channel_dim, kernel_size=1, activation=nn.ReLU)

    def forward(self, features, audio_embed):
        """
        implementation of complex numbers multiplication
        using (a + bi)(c + di) = ac - bd + i(ad + bc)
        """
        b, c, t, f = features.shape
        m = self.conv(self.prelu(features))
        m = m.view(b // self.n_speakers, self.n_speakers, c, t, f)
        m_real = m[:, :, : c // 2]
        m_imag = m[:, :, c // 2 :]
        audio_embed = audio_embed[:, None, ...]
        audio_embed_real = audio_embed[:, :, : c // 2]
        audio_embed_imag = audio_embed[:, :, c // 2 :]
        output_real = m_real * audio_embed_real - m_imag * audio_embed_imag
        output_imagen = m_real * audio_embed_imag + m_imag * audio_embed_real
        return torch.cat((output_real, output_imagen), dim=2)  # b x n_spk x c x t f
