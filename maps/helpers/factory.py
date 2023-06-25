#!/usr/bin/env python3
from typing import Callable, Optional
from torch import nn
from maps.helpers.initializers import ortho_init


class Factory:
    @staticmethod
    def create_pi(state_dim, act_dim, policy_head, hidden_dim=128, initializer: Optional[Callable] = ortho_init):
        pi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            policy_head
        )
        if initializer is not None:
            initializer(pi[0], gain=1)
            initializer(pi[2], gain=1)
            initializer(pi[4], gain=1e-2)
        return pi

    @staticmethod
    def create_vfn(state_dim, mean_and_var=False, hidden_dim=256, initializer: Optional[Callable] = ortho_init, activation = nn.Tanh):
        output_dim = 2 if mean_and_var else 1
        vfn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )
        if initializer is not None:
            initializer(vfn[0], gain=1)
            initializer(vfn[2], gain=1)
            initializer(vfn[4], gain=1)
            # initializer(vfn[4], gain=10)
        return vfn

    @staticmethod
    def create_state_action_nn(state_dim, action_dim, hidden_dim=256, initializer: Optional[Callable] = ortho_init, activation = nn.Tanh):
        state_nn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, state_dim * 2),
        )
        if initializer is not None:
            initializer(state_nn[0], gain=1)
            initializer(state_nn[2], gain=1)
            initializer(state_nn[4], gain=1)
            # initializer(vfn[4], gain=10)
        return state_nn

    @staticmethod
    def create_state_nn(state_dim, hidden_dim=256, initializer: Optional[Callable] = ortho_init, activation = nn.Tanh):
        state_nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, state_dim * 2),
        )
        if initializer is not None:
            initializer(state_nn[0], gain=1)
            initializer(state_nn[2], gain=1)
            initializer(state_nn[4], gain=1)
            # initializer(vfn[4], gain=10)
        return state_nn
