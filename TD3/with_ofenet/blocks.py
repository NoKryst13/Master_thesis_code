import sys
import torch as th
import torch.nn as nn
from typing import Type, Union


class DensenetBlock(nn.Module):
    """
    Create DensenetBlock with arbitrary input-size -> LazyLinear
    Batch Normalization is included
    """
    def __init__(
        self,
        units_per_layer: int,
    ):
        super(DensenetBlock, self).__init__()
        self.fc = nn.LazyLinear(units_per_layer)
        self.silu = nn.SiLU()
        self.normalizer = nn.BatchNorm1d(units_per_layer)

    def forward(self, x):
        identity_map = x

        x = self.fc(x)

        x = self.normalizer(x)

        x = self.silu(x)
        x = th.cat([x, identity_map], 1)
        return x
