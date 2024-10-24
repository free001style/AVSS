import torch
import torch.nn as nn

from src.model.layers.caf import CAFBlock
from src.model.layers.rtfs_block import RTFSBlock
from src.model.layers.vp_block import VPBlock


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        channel_dim,
        video_embed_dim,
        n_head=4,
        R=12,
        hidden_dim=64,
        freqs=128,
        q_audio=2,
        q_video=4,
        n_speakers=2,
    ):
        super(SeparationNetwork, self).__init__()
        self.R = R
        self.n_speakers = n_speakers
        rtfs_block = RTFSBlock(channel_dim, hidden_dim, freqs, q_audio)
        self.vp = VPBlock(q_video, video_embed_dim, hidden_dim)
        self.ap = rtfs_block
        self.fusion = CAFBlock(
            channel_dim_a=channel_dim,
            channel_dim_v=video_embed_dim,
            h=n_head,
            n_speakers=n_speakers,
        )
        self.rtfs_blocks = rtfs_block

    def forward(self, audio_embed, video_embed):
        b, c, t, f = audio_embed.shape
        residual = audio_embed
        audio_embed = self.ap(audio_embed)
        # video_embed = self.vp(video_embed)
        # fused = self.fusion(audio_embed, video_embed)
        fused = audio_embed

        for i in range(self.R):
            if i > 0:
                # fused = fused.view(b, self.n_speakers, c, t, f)
                # fused += residual[:, None, ...]
                # fused = fused.view(b * self.n_speakers, c, t, f)
                fused += residual
            fused = self.rtfs_blocks(fused)
        return fused
