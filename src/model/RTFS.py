import torch
import torch.nn as nn

from .layers.separation_network import SeparationNetwork
from .layers.audio_enc import AudioEncoder
from .layers.audio_dec import AudioDecoder


class RTFS(nn.Module):
    def __init__(self):
        super(RTFS, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        self.video_encoder = nn.Sequential()
        self.separator = SeparationNetwork()
        self.mask = nn.Sequential()

    def forward(self, mix, video, **batch):
        audio_embed = self.audio_encoder(mix)
        video_embed = self.video_encoder(video)
        features = self.separator(audio_embed, video_embed)
        masked_features = self.mask(audio_embed, features)
        predict = self.audio_decoder(masked_features)
        return {"predict": predict}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
