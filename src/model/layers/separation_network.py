import torch
import torch.nn as nn


class SeparationNetwork(nn.Module):
    def __init__(self):
        super(SeparationNetwork, self).__init__()
        self.vp = nn.Sequential()
        self.ap = nn.Sequential()
        self.fusion = nn.Sequential()
        self.rtfs_blocks = nn.Sequential()

    def forward(self, audio_embed, video_embed):
        audio_embed = self.ap(audio_embed)
        video_embed = self.vp(video_embed)
        fused = self.fusion(audio_embed, video_embed)
        return self.rtfs_blocks(fused)
