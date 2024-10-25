import torch
import torch.nn as nn

from src.model.videonet.resnet import BasicBlock, ResNet


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Lipreading(nn.Module):
    def __init__(self):
        super(Lipreading, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type="swish")
        frontend_relu = nn.SiLU()

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = self.frontend3D(x)  # b x c' x t' x h x w
        t_new = x.shape[2]
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(b, t_new, x.size(1)).transpose(
            1, 2
        )  # b x embed_dim x t'
        return x
