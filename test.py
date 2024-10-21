# m = RTFS(R=1)
# x = torch.randn(5, 32000)
# y = torch.randn(5, 2, 50, 96, 96)
# print(m(x, y)['predict'].shape)
import os

import torch
import torch.nn as nn
from dotenv import load_dotenv

from src import *

# for testing
from src.model import RTFS

load_dotenv()

MY_ENV_VAR = os.getenv("KEY")

print(MY_ENV_VAR)
