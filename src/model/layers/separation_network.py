import torch
import torch.nn as nn
from src.model.layers import CAFBlock
from src.model.layers import RTFSBlock


class SeparationNetwork(nn.Module):
    def __init__(self, cin_a, cin_v, h, R=12):
        super(SeparationNetwork, self).__init__()
        self.R = R
        rtfs_block = RTFSBlock(256, 64, 128, 2)
        self.vp = nn.Sequential()
        self.ap = rtfs_block
        self.fusion = CAFBlock(cin_a=cin_a, cin_v=cin_v, h=h)
        self.rtfs_blocks = rtfs_block

    def forward(self, audio_embed, video_embed):
        residual = audio_embed
        audio_embed = self.ap(audio_embed)
        video_embed = self.vp(video_embed)
        fused = self.fusion(audio_embed, video_embed)
        for i in range(self.R):
            if i > 0:
                fused += residual
            fused = self.rtfs_blocks(fused)
        return fused
