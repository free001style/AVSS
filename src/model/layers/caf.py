import torch
from torch import nn
import torch.nn.functional as F


class CAFBlock(nn.Module):
    def __init__(self, cin_a, cin_v, h, eps=1e-5):
        super(CAFBlock, self).__init__()
        self.eps = eps
        self.h = h
        self.cin_a = cin_a
        self.conv_gate = nn.Conv2d(cin_a, cin_a, kernel_size=(1, 1), groups=cin_a, bias=False)
        self.conv_val = nn.Conv2d(cin_a, cin_a, kernel_size=(1, 1), groups=cin_a, bias=False)
        self.gln_gate = nn.BatchNorm2d(cin_a)
        self.gln_val = nn.BatchNorm2d(cin_a)

        self.conv_attn = nn.Conv1d(cin_v, cin_a * self.h, kernel_size=(1,), groups=cin_a, bias=False)
        self.conv_key = nn.Conv1d(cin_v, cin_a, kernel_size=(1,), groups=cin_a, bias=False)
        self.gln_attn = nn.GroupNorm(num_groups=1, num_channels=cin_a * self.h, eps=self.eps)
        self.gln_key = nn.GroupNorm(num_groups=1, num_channels=cin_a, eps=self.eps)

    def forward(self, a1, v1):
        # a1 = (b, Ca, Ta, F) v1 = (b, Cv, Tv)
        b, _, time, _ = a1.shape
        v1 = self.gln_attn(self.conv_attn(v1))  # (b, cin_a*h, Tv)
        vm = torch.mean(v1.reshape(b, self.cin_a, self.h, -1), dim=2, keepdim=False).view(b, self.cin_a, -1)
        v_attn = F.interpolate(F.softmax(vm, -1), size=time, mode='nearest')
        v_key = F.interpolate(self.gln_key(self.conv_key(v1), size=time, mode='nearest'))

        a_val = self.gln_val(self.conv_val(a1))
        a_gate = F.relu(self.gln_gate(self.conv_gate(a1)))

        f1 = v_attn[:, :, :, None] * a_val
        f2 = v_key[:, :, :, None] * a_gate
        return f1 + f2  # (Ca, Ta, F)
