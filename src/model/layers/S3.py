import torch
from torch import nn

from src.model.layers.conv import Conv


class S3(nn.Module):
    """
    Spectral Source Separation module:
        1) predict mask from features
        2) masking audio embeddings using complex multiplication.
    """

    def __init__(self, channel_dim):
        """
        Args:
            channel_dim (int): audio channel dimension (C_a in paper).
        """
        super(S3, self).__init__()
        self.prelu = nn.PReLU()
        self.conv = Conv(channel_dim, channel_dim, kernel_size=1, activation=nn.ReLU)

    def forward(self, features, audio_embed):
        """
        Args:
            features (Tensor): (B, C_a, T_a, F) audio features after RTFS blocks.
            audio_embed (Tensor): (B, C_a, T_a, F) audio embeddings from Audio Encoder.
        Returns:
            predict (Tensor): (B, C_a, T_a, F) masked audio embeddings.
        """
        b, c, t, f = features.shape
        m = self.conv(self.prelu(features))
        m_real = m[:, : c // 2]
        m_imag = m[:, c // 2 :]
        audio_embed_real = audio_embed[:, : c // 2]
        audio_embed_imag = audio_embed[:, c // 2 :]
        output_real = m_real * audio_embed_real - m_imag * audio_embed_imag
        output_imagen = m_real * audio_embed_imag + m_imag * audio_embed_real
        return torch.cat((output_real, output_imagen), dim=1)
