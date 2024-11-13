import torch
import torch.nn as nn

from src.model.layers.caf import CAFBlock
from src.model.layers.rtfs_block import RTFSBlock
from src.model.layers.vp_block import VPBlock


class SeparationNetwork(nn.Module):
    """
    Separation Network module:
        1) Preprocessing audio embeddings
        2) Preprocessing video embeddings
        3) Fusion audio and video embeddings
        4) R-stacked RTFS blocks for fused features.
    """

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
        use_video=True,
    ):
        """
        Args:
            channel_dim (int): audio channel dimension (C_a in paper).
            video_embed_dim (int): video embedding dimension (C_v in paper).
            n_head (int): number of heads in attention fusion.
            R (int): number of RTFS blocks (with sharing their weights).
            hidden_dim (int): channel dimension in RTFS block (D in paper).
            freqs (int): number of frequencies (needs for cfLN).
            q_audio (int): number of spacial dim decreasing in compression phase for audio (q in paper).
            q_video (int): number of spacial dim decreasing in compression phase for video (q in paper).
            use_video (bool): whether to use video of speakers.
        """
        super(SeparationNetwork, self).__init__()
        self.use_video = use_video
        self.R = R
        rtfs_block = RTFSBlock(channel_dim, hidden_dim, freqs, q_audio)
        if use_video:
            self.vp = VPBlock(q_video, video_embed_dim, hidden_dim)
        self.ap = rtfs_block
        self.fusion = CAFBlock(
            channel_dim_a=channel_dim,
            channel_dim_v=video_embed_dim,
            h=n_head,
        )
        self.rtfs_blocks = rtfs_block

    def forward(self, audio_embed, video_embed):
        """
        Args:
            audio_embed (Tensor): (B, C_a, T_a, F) audio embedding.
            video_embed (Tensor): (B, C_v, T_v) video embedding.
        Returns:
            fused (Tensor): (B, C_a, T_a, F)
        """
        residual = audio_embed
        audio_embed = self.ap(audio_embed)
        if self.use_video:
            video_embed = self.vp(video_embed)
            fused = self.fusion(audio_embed, video_embed)
        else:
            fused = audio_embed.clone()
        for i in range(self.R):
            if i > 0:  # TODO
                fused += residual
            fused = self.rtfs_blocks(fused)
        return fused
