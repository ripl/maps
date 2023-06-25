#!/usr/bin/env python3
from torch import nn

def ortho_init(layer, gain, zeros_bias: bool = True):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if zeros_bias:
        nn.init.zeros_(layer.bias)

def xavier_init(layer, gain, zeros_bias: bool = True):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if zeros_bias:
        nn.init.zeros_(layer.bias)
