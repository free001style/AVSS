from src import *
import torch
import torch.nn as nn

# for testing
from src.model import RTFS

m = RTFS(0)
x = torch.randn(5, 32000)
y = torch.randn(5, 512, 500)
print(m(x, y))