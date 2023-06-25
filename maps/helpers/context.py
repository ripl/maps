#!/usr/bin/env python3
"""Copied from pfnet/pfrl"""

from torch import nn
from contextlib import contextmanager


@contextmanager
def evaluating(net: nn.Module):
    """Temporarily switch the nn.Module to evaluation mode."""
    if isinstance(net, nn.Module):
        istrain = net.training
        try:
            net.eval()
            yield net
        finally:
            if istrain:
                net.train()
    else:
        # Do nothing
        yield
