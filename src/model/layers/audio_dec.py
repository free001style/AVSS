import torch
import torch.nn as nn


class AudioDecoder(nn.Module):
    def __init__(self, cin_a=256, win=256, hop_length=128, n_speak=2):
        super(AudioDecoder, self).__init__()
        self.n_speak = n_speak
        self.win = win
        self.hop_length = hop_length
        self.conv = nn.ConvTranspose2d(cin_a, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.window = torch.hann_window(self.win)

    def forward(self, x, length):
        b, n_speak, ch, t, f = x.shape
        x = self.conv(x.view(b*n_speak, ch, t, f))  # (b, ch, time, freq)
        x = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2)
        audio = torch.istft(x, n_fft=self.win, hop_length=self.hop_length,
                            window=self.window.to(x.device), length=length)  # (b, length)
        return audio.view(b, n_speak, length)
