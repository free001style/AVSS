import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, cin_a=256, win=256, hop_length=128):
        super(AudioEncoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.conv = nn.Conv2d(2, cin_a, kernel_size=(3, 3), padding=(1, 1), bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.gln = nn.GroupNorm(num_groups=1, num_channels=cin_a, eps=1e-5)
        self.window = torch.hann_window(self.win)

    def forward(self, audio):
        alpha = torch.stft(audio, n_fft=self.win, hop_length=self.hop_length,
                           window=self.window.to(audio.device), return_complex=False)  # (b, time, freq, 2)
        alpha = self.conv(alpha.permute(0, 3, 2, 1))  # (b, cin_a, time, freq)
        alpha = F.relu(self.gln(alpha))
        return alpha
