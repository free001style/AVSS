import torch

from torch import nn
from src.model.layers.conv import Conv


class S3(nn.Module):
    def __init__(self, cin_a):
        super(S3, self).__init__()
        self.prelu = nn.PReLU()
        self.conv = Conv(cin_a, cin_a, kernel_size=(1,))
        self.relu = nn.ReLU()

    def forward(self, aR, a0):
        '''
        implementation of complex numbers multiplication
        using (a + bi)(c + di) = ac - bd + i(ad + bc)
        '''
        b, c, t, f = aR.shape
        m = self.relu(self.conv(self.prelu(aR)))
        m_real = m[:, :c//2, :, :]
        m_imagine = m[:, c//2:, :, :]
        a0_real = a0[:, :c//2, :, :]
        a0_imagen = a0[:, c//2:, :, :]
        output_real = m_real * a0_real - m_imagine * a0_imagen
        output_imagen = m_real * a0_imagen + m_imagine * a0_real

        return torch.cat((output_real, output_imagen), dim=1)
