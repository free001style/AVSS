import torch
import torch.nn as nn

from src.model.layers.audio_enc_dec import AudioDecoder, AudioEncoder
from src.model.layers.S3 import S3
from src.model.layers.separation_network import SeparationNetwork
from src.model.videonet.lipreading import Lipreading


class RTFS(nn.Module):
    """
    RTFS-Net implementation based on https://arxiv.org/abs/2309.17189.
    """

    def __init__(
        self,
        R=12,
        n_speakers=2,
        channel_dim=256,
        n_fft=256,
        hop_length=128,
        video_embed_dim=512,
        fusion_n_head=4,
        hidden_dim=64,
        freqs=128,
        q_audio=2,
        q_video=4,
        lipreading_model_path="./data/other/lipreading_model.pth",
        use_video=True,
    ):
        """
        Args:
            R (int): number of RTFS blocks (with sharing their weights).
            n_speakers (int): number of speakers in mix for separation.
            channel_dim (int): audio channel dimension (C_a in paper).
            n_fft (int): n_fft for stft and istft.
            hop_length (int): hop_length for stft and istft.
            video_embed_dim (int): video embedding dimension (C_v in paper).
            fusion_n_head (int): number of heads in attention fusion.
            hidden_dim (int): channel dimension in RTFS block (D in paper).
            freqs (int): number of frequencies (needs for cfLN).
            q_audio (int): number of spacial dim decreasing in compression phase for audio (q in paper).
            q_video (int): number of spacial dim decreasing in compression phase for video (q in paper).
            lipreading_model_path (str): path for pretrained lipreading model.
            use_video (bool): whether to use video of speakers.
        """
        super(RTFS, self).__init__()
        self.use_video = use_video
        self.video_embed_dim = video_embed_dim
        self.n_speakers = n_speakers
        self.audio_encoder = AudioEncoder(
            channel_dim=channel_dim, n_fft=n_fft, hop_length=hop_length
        )
        self.audio_decoder = AudioDecoder(
            channel_dim=channel_dim,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        if self.use_video:
            self.video_encoder = Lipreading()
            self.video_encoder.load_state_dict(
                torch.load(lipreading_model_path, weights_only=True)
            )
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
            use_video=use_video,
        )
        self.mask = S3(channel_dim=channel_dim)

    def forward(self, mix, video=None, **batch):
        """
        Args:
            mix (Tensor): (B, L) tensor of mix audio.
            video (Tensor): (B, n_spk, T, H, W) tensor of sequence of lips frames, if video isn't used then None.
        Returns:
            predict (dict): dict with key predict: (B, n_spk, length) -- separated audio for each speaker.
        """
        b, l = mix.shape
        audio_embed = self.audio_encoder(mix)
        if self.use_video:
            video_embed = self.video_encoder(video).view(
                audio_embed.shape[0], self.n_speakers, self.video_embed_dim, -1
            )
        masked = []
        for i in range(self.n_speakers):
            if self.use_video:
                features = self.separator(audio_embed, video_embed[:, i])
            else:
                features = self.separator(audio_embed, None)
            masked_embed = self.mask(features, audio_embed)
            masked.append(masked_embed)
        predict = (
            self.audio_decoder(torch.cat(masked), l)
            .view(self.n_speakers, b, l)
            .transpose(0, 1)
        )
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
