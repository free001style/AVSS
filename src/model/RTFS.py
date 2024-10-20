import torch
import torch.nn as nn

from src.model.layers.audio_enc_dec import AudioDecoder, AudioEncoder
from src.model.layers.separation_network import SeparationNetwork
from src.model.layers.video_enc import get_video_model


class RTFS(nn.Module):
    def __init__(
        self,
        video_enc_config,
        channel_dim=256,
        win_length=255,
        hop_length=128,
        n_speakers=2,
        video_embed_dim=512,
        fusion_n_head=4,
        R=12,
        hidden_dim=64,
        freqs=128,
        q_audio=2,
        q_video=4,
    ):
        super(RTFS, self).__init__()
        self.audio_encoder = AudioEncoder(
            channel_dim=channel_dim, win_length=win_length, hop_length=hop_length
        )
        self.audio_decoder = AudioDecoder(
            channel_dim=channel_dim,
            win_length=win_length,
            hop_length=hop_length,
            n_speakers=n_speakers,
        )
        self.video_encoder = get_video_model(video_enc_config)
        self.separator = SeparationNetwork(
            channel_dim,
            video_embed_dim,
            fusion_n_head,
            R,
            hidden_dim=hidden_dim,
            freqs=freqs,
            q_audio=q_audio,
            q_video=q_video,
        )
        self.mask = nn.Sequential()

    def forward(self, mix, video, **batch):
        audio_embed = self.audio_encoder(mix)
        video_embed = self.video_encoder(video)
        features = self.separator(audio_embed, video_embed)
        masked_features = self.mask(audio_embed, features)
        predict = self.audio_decoder(masked_features, mix.shape[-1])
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


m = RTFS(0)
x = torch.randn(5, 32000)
y = torch.randn(5, 512, 500)
print(m(x, y))
