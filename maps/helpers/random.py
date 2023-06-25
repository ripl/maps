#!/usr/bin/env python3
"""Copied from pfnet/pfrl"""
import random

import numpy as np
import torch


def set_random_seed(seed):
    """Set a given random seed to Pytorch's random number generator

    torch.manual_seed() seeds the RNG for all devices (both CPU and CUDA)

    See https://pytorch.org/docs/stable/notes/randomness.html for more details

    Args:
        seed (int): Random seed [0, 2 ** 32).
    """
    # PFRL depends on random
    random.seed(seed)
    # PFRL depends on numpy.random
    np.random.seed(seed)
    # torch.manual_seed is enough for the CPU and GPU
    torch.manual_seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
