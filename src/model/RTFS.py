import torch
import torch.nn as nn

from src.model.layers.audio_enc_dec import AudioDecoder, AudioEncoder
from src.model.layers.S3 import S3
from src.model.layers.separation_network import SeparationNetwork
from src.model.videonet.lipreading import Lipreading


class RTFS(nn.Module):
    def __init__(
        self,
        channel_dim=256,
        win_length=255,
        hop_length=128,
        # n_speakers=2,
        video_embed_dim=512,
        fusion_n_head=4,
        R=12,
        hidden_dim=64,
        freqs=128,
        q_audio=2,
        q_video=4,
        lipreading_model_path="./data/other/lipreading_model.pth",
    ):
        super(RTFS, self).__init__()
        self.audio_encoder = AudioEncoder(
            channel_dim=channel_dim, win_length=win_length, hop_length=hop_length
        )
        self.audio_decoder = AudioDecoder(
            channel_dim=channel_dim,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.video_encoder = Lipreading()
        # self.video_encoder.load_state_dict(
        #     torch.load(lipreading_model_path, weights_only=True)
        # )
        self.video_encoder.eval().requires_grad_(False)
        self.separator = SeparationNetwork(
            channel_dim,
            video_embed_dim,
            fusion_n_head,
            R,
            hidden_dim=hidden_dim,
            freqs=freqs,
            q_audio=q_audio,
            q_video=q_video,
            # n_speakers=n_speakers,
        )
        self.mask = S3(channel_dim=channel_dim)

    def forward(self, mix, video, **batch):
        """
        mix b x L
        video b x 2 x t x h x w
        """
        audio_embed = self.audio_encoder(mix)
        predicts = []
        for target_video_num in range(2):
            video_embed = self.video_encoder(video[:, target_video_num][:, None, ...])
            features = self.separator(audio_embed, video_embed)
        # features = self.separator(audio_embed, None)

            masked_features = self.mask(features, audio_embed)
            predicts.append(self.audio_decoder(masked_features, mix.shape[-1]))

        return {"predict": torch.stack(predicts, dim=0).transpose(1, 0)}

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
