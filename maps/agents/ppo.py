#!/usr/bin/env python3
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from maps import logger
from maps.helpers import to_torch
from maps.helpers.data import yield_batch_infinitely
from torch import nn
from torch.nn import functional as F

from .base import Agent


def calc_loss_actor(pi: nn.Module, obs: torch.Tensor, action: torch.Tensor, tgt_val: torch.Tensor, log_probs_old: torch.Tensor, clip_eps: float = 0.2):
    """PPO loss (clipped policy probability ratio)"""

    # HACK: Get around annoying nan issue
    if log_probs_old.isnan().any():
        logger.warn('Nan detected in log_probs_old (calc_loss_actor)')
        return

    # HACK: Stabilize
    log_probs_old = torch.clip(log_probs_old, -50, 50)

    print(f'log_probs_old min: {log_probs_old.min()},\tmax: {log_probs_old.max()}')
    pi_ratio = torch.exp(pi(obs).log_prob(action) - log_probs_old)
    print(f'pi_ratio min: {pi_ratio.min()},\tmax: {pi_ratio.max()}')
    clipped_pi_ratio = torch.min(
            pi_ratio * tgt_val,
            torch.clamp(pi_ratio, 1 - clip_eps, 1 + clip_eps) * tgt_val,
    )
    return - clipped_pi_ratio.mean()


def calc_loss_critic(vfn: nn.Module, obs: torch.Tensor, tgt_val: torch.Tensor):
    # Value loss is a simple L2
    return F.mse_loss(vfn(obs), tgt_val)


def update_critic_ensemble(agent: Agent, transitions: Sequence[dict], num_epochs: int = 10, batch_size: int = 128, std_from_means: bool = False):
    from .mamba import ValueEnsemble
    assert 'v_teacher' in transitions[0], 'Make sure to attach v_teacher to transitions by calling _add_advantage_and_value_target_to_episode function.'
    assert isinstance(agent.vfn, ValueEnsemble)

    sample_itr = yield_batch_infinitely(transitions, batch_size=batch_size)
    losses = []

    num_updates = max(len(transitions) // batch_size, 1) * num_epochs
    for _ in range(num_updates):
        # Compute loss against each value NN in ValueEnsemble
        loss_per_itr = 0
        for vfn_idx in range(agent.vfn.num_value_nns):
            batch = next(sample_itr)
            states = batch['state'].type(torch.float32)
            if agent.obs_normalizer:
                states = agent.obs_normalizer(states, update=False)
            vs_teacher = batch['v_teacher'].type(torch.float32)

            # Same shape as vs_pred: (batch_size, 1)
            vs_teacher = vs_teacher[..., None]

            distr = agent.vfn.forward_single(states, vfn_idx)
            if std_from_means:
                loss_critic = F.mse_loss(distr.mean, vs_teacher)
            else:
                loss_critic = -distr.log_prob(vs_teacher).mean()

            # NOTE: It's critical to have set_to_none=True
            # Otherwise, it updates other value NNs as well, which causes an error in the end.
            agent.optimizer.zero_grad(set_to_none=True)
            loss_critic.backward()
            agent.optimizer.step()

            loss_per_itr += loss_critic.item()
        losses.append(loss_per_itr / agent.vfn.num_value_nns)

    return np.mean(losses), losses


def update_critic(agent: Agent, transitions: Sequence[dict], num_epochs: int = 10, batch_size: int = 128):
    assert 'v_teacher' in transitions[0], 'Make sure to attach v_teacher to transitions by calling _add_advantage_and_value_target_to_episode function.'

    if hasattr(agent, 'batch_size'):
        assert agent.batch_size == batch_size

    sample_itr = yield_batch_infinitely(transitions, batch_size)
    losses = []

    num_updates = max(len(transitions) // batch_size, 1) * num_epochs
    for _ in range(num_updates):
        batch = next(sample_itr)
        states = batch['state'].type(torch.float32)
        if agent.obs_normalizer:
            states = agent.obs_normalizer(states, update=False)
        vs_teacher = batch['v_teacher'].type(torch.float32)

        # Same shape as vs_pred: (batch_size, 1)
        vs_teacher = vs_teacher[..., None]

        agent.optimizer.zero_grad()
        loss_critic = calc_loss_critic(agent.vfn, states, vs_teacher)
        loss_critic.backward()
        agent.optimizer.step()
        losses.append(loss_critic.item())

    return np.asarray(losses).mean()


class PPOAgent(Agent):
    def __init__(self, pi: nn.Module, vfn: nn.Module, optimizer, obs_normalizer: Callable | None = None, max_grad_norm: None | float = None, standardize_advantages: bool = True,
                 gamma: float = 0.995, lambd: float = 0.97) -> None:
        super().__init__(pi, vfn, optimizer, obs_normalizer)
        self.max_grad_norm = max_grad_norm
        self.standardize_advantages = standardize_advantages

        self.coef_critic = 1.
        self.coef_entropy = 0.01

        self.gamma = gamma #XL: not used
        self.lambd = lambd #XL: not used
        self._load_path = ''

    def save(self, save_path):
        print(f'saving model to {str(save_path)}...')
        model_dict = {'pi': self.pi.state_dict(),
                      'vfn': self.vfn.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'obs_normalizer': self.obs_normalizer}

        torch.save(model_dict, save_path)
        print(f'saving model to {str(save_path)}...done')

    def load(self, load_path):
        print(f'loading model from {str(load_path)}...')
        model_dict = torch.load(load_path)
        self.pi.load_state_dict(model_dict['pi'])
        self.vfn.load_state_dict(model_dict['vfn'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.obs_normalizer = model_dict['obs_normalizer']
        self._load_path = load_path
        print(f'loading model from {str(load_path)}...done')

    @property
    def _step(self):
        fname_stem = Path(self._load_path).stem
        try:
            return int(fname_stem.lstrip('step_'))
        except ValueError as e:
            print('captured', e)
            print(f'Failed to convert the filename {fname_stem} to an int')
            return fname_stem

    def update(self, transitions, num_epochs=10, batch_size: int = 128):
        """Update the actor and critic with `transitions` that have `adv`, `log_prob`, `v_pred` and `v_teacher` already attached."""
        if self.standardize_advantages:
            advs = [tr['adv'] for tr in transitions]
            std_advs, mean_advs = torch.std_mean(to_torch(advs), unbiased=False)

        sample_itr = yield_batch_infinitely(transitions, batch_size=batch_size)
        loss_dict = defaultdict(list)

        num_updates = max(len(transitions) // batch_size, 1) * num_epochs

        for epoch in range(num_updates):
            batch = next(sample_itr)

            states = batch['state'].type(torch.float32)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            actions = batch['action'].type(torch.float32)
            distribs = self.pi(states)
            # vs_pred = self.critic.value_nn(states)

            advs = batch['adv'].type(torch.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = batch['log_prob'].type(torch.float32)
            vs_pred_old = batch['v_pred'].type(torch.float32)
            vs_teacher = batch['v_teacher'].type(torch.float32)

            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.optimizer.zero_grad()
            loss_actor = calc_loss_actor(self.pi, states, actions, advs, log_probs_old)
            if loss_actor is None:
                loss_actor = 0

            loss_critic = calc_loss_critic(self.vfn, states, vs_teacher)
            loss_entropy = -torch.mean(distribs.entropy())
            loss = (
                loss_actor
                + self.coef_critic * loss_critic
                + self.coef_entropy * loss_entropy
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.pi.parameters()) + list(self.vfn.parameters()), self.max_grad_norm
                )
            self.optimizer.step()

            # Append to loss dict
            loss_dict['actor'].append(loss_actor.item())
            loss_dict['critic'].append(loss_critic.item())
            loss_dict['entropy'].append(loss_entropy.item())
            loss_dict['all'].append(loss.item())

            self.n_updates += 1

        loss_info = {
            f'loss/{key}': np.asarray(arr).mean() for key, arr in loss_dict.items()
        }

        return loss_info
