#!/usr/bin/env python3
from __future__ import annotations

from os import PathLike
from typing import Callable

import numpy as np
import torch
from torch import nn


class Agent:
    def __init__(self, pi: nn.Module, vfn: nn.Module, optimizer: torch.optim.Optimizer, obs_normalizer: Callable | None = None, device='cuda', *args, **kwargs) -> None:
        self.pi = pi
        self.vfn = vfn
        self.optimizer = optimizer
        self.device = device
        self.n_updates = 0

        self.obs_normalizer = obs_normalizer

    @torch.no_grad()
    def act(self, state, mode=False) -> np.ndarray:
        # Add batch dim, send to cuda
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.obs_normalizer:
            state = self.obs_normalizer(state, update=False)

        distr = self.pi(state)
        if not mode:
            action = distr.sample()
        else:
            action = distr.mode
        return action.cpu().numpy().squeeze(0)

    def to(self, device):
        self.pi.to(device)
        self.vfn.to(device)
        if hasattr(self, 'obs_normalizer'):
            self.obs_normalizer.to(device)

    def update(self, *args, **kwargs) -> torch.TensorType:
        raise NotImplementedError()

    def save(self, path: str | PathLike):
        raise NotImplementedError()

    def load(self, path: str | PathLike):
        raise NotImplementedError()

    def train(self) -> None:
        self.pi.train()
        self.vfn.train()

    def eval(self) -> None:
        self.pi.eval()
        self.vfn.eval()
