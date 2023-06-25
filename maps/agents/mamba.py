#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributions as torchd
from maps.helpers import to_torch
from maps.helpers.data import yield_batch_infinitely
from torch import nn
from torch.nn import functional as F

from .base import Agent
from .ppo import calc_loss_actor


class ValueEnsemble(nn.Module):
    """Maintain `num_value_nn` networks as an ensemble."""
    def __init__(self, make_vfn: Callable, num_value_nns=5, obs_normalizers=None, var_func=F.softplus, beta=1.0, std_from_means=False) -> None:
        """
        - beta (float): Adjust the UCB bound stddev (i.e., range is `mean + beta * std`)
        """
        super().__init__()
        self.var_func = var_func
        self.vfns = nn.ModuleList([
            make_vfn()
            for _ in range(num_value_nns)
        ])

        if obs_normalizers is not None:
            assert len(obs_normalizers) == len(self.vfns)
            self.obs_normalizers = obs_normalizers
        else:
            self.obs_normalizers = [lambda obs, update: obs for _ in self.vfns]

        self.beta = beta
        self.num_value_nns = num_value_nns

        self.stddev_coef = 1.0
        self.std_from_means = std_from_means

    @staticmethod
    def compute_stats(distributions: List[torchd.Distribution], beta=1.0, name='value_stats', stddev_coef: float = 1.0, std_from_means: bool = False):
        """Given the list of distributions, compute statistics and return them."""
        from collections import namedtuple
        stats = namedtuple(name, 'mean std upper lower all_means all_vars max_gap upper_max_gap lower_max_gap')

        # NOTE: How to aggregate predicted variances? --> (https://arxiv.org/pdf/1612.01474.pdf; Right before Section 3)
        all_means = torch.stack([distr.mean for distr in distributions], dim=0)
        mean = all_means.mean(0)

        if std_from_means:
            var = torch.var(all_means, dim=0)
            all_vars = torch.stack([var for _ in distributions], dim=0)
        else:
            all_vars = torch.stack([distr.variance for distr in distributions], dim=0)
            var = torch.stack([(distr.mean ** 2 + distr.variance) for distr in distributions], dim=0).mean(0) - mean ** 2
            # var = (stddev_coef ** 2) * torch.var(all_means, dim=0)
            var = (stddev_coef ** 2) * var
        std = torch.sqrt(var)

        upper = mean + beta * std
        lower = mean - beta * std

        max_gap = torch.max(all_means, dim=0)[0]- torch.min(all_means, dim=0)[0]
        upper_max_gap = mean + max_gap
        lower_max_gap = mean - max_gap

        return stats(mean=mean, std=std, upper=upper, lower=lower,
                     all_means=all_means, all_vars=all_vars,
                     max_gap=max_gap, upper_max_gap=upper_max_gap,lower_max_gap=lower_max_gap)

    def forward_all(self, obs, normalize_input: bool = False) -> List[torchd.Distribution]:
        """Run forward on all networks and returns the distributions."""
        distributions = [self.forward_single(obs, vfn_idx, normalize_input=normalize_input) for vfn_idx in range(self.num_value_nns)]
        return distributions

    def forward_single(self, obs, vfn_idx, normalize_input=False):
        """Run forward on a single network specified by nn_idx."""
        assert len(obs.shape) == 2, f'obs has invalid shape: {obs.shape}'

        if normalize_input:
            normalizer = self.obs_normalizers[vfn_idx]
        else:
            normalizer = lambda obs, update: obs

        vfn = self.vfns[vfn_idx]
        mean_and_var = vfn(normalizer(obs, update=False))
        mean, pre_var = torch.chunk(mean_and_var, 2, dim=1)

        var = self.var_func(pre_var) + 1e-8
        # var = torch.exp(log_scale * 2)
        dist = torchd.Independent(
            torchd.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        return dist

    def forward(self, obs, normalize_input: bool = False):
        distributions = self.forward_all(obs, normalize_input=normalize_input)
        # return torch.stack([distr.mean for distr in distributions], dim=0).mean(0)
        stacked = torch.stack([distr.mean for distr in distributions], dim=0)
        # print('stacked', stacked.shape)
        return stacked.mean(0)

    def forward_stats(self, obs, normalize_input: bool = False):
        values = self.forward_all(obs, normalize_input=normalize_input)
        stats = self.compute_stats(values, beta=self.beta, stddev_coef=self.stddev_coef, std_from_means=self.std_from_means)
        return stats


class MaxValueFn(nn.Module):
    """Maintain multiple value functions, and the `forward` returns the maximum value of them."""
    def __init__(self, value_fns: List[nn.Module], obs_normalizers=None) -> None:
        super().__init__()
        self.value_fns = value_fns

        if obs_normalizers is not None:
            assert len(obs_normalizers) == len(value_fns)
            self.obs_normalizers = obs_normalizers
        else:
            self.obs_normalizers = [lambda obs, update: obs for _ in value_fns]

    def forward(self, x, normalize_input=False):
        # x.shape: torch.Size([batch_size, obs_dim])
        if normalize_input:
            normalizers = self.obs_normalizers
        else:
            normalizers = [lambda obs, update: obs for _ in self.value_fns]

        values = torch.stack([vfn(normalizer(x, update=False)) for normalizer, vfn in zip(normalizers, self.value_fns)], dim=-1)
        max_obj = torch.max(values, dim=-1)
        return max_obj.values


class ActiveStateExplorer:
    def __init__(self, value_fns: List[ValueEnsemble], sigma, uncertainty="std") -> None:
        self.value_fns = value_fns
        self.sigma = sigma
        self._uncertainty = uncertainty

    def _get_best_expert(self, obs):
        # Find the value function whose upper bound is the best
        sorted_pairs = sorted([(idx, vfn.forward_stats(obs)) for idx, vfn in enumerate(self.value_fns)], key=lambda x: x[1].upper)
        best_idx, best_valobj = sorted_pairs[-1]

        return best_idx, best_valobj

    @torch.no_grad()
    def should_explore(self, obs):
        """When a (best) expert on current state is sure about what will happen, we will not gain anything by switching to expert,
        thus we should 'explore', by continuing on running the learner policy.
        """
        obs = to_torch(obs).unsqueeze(0)
        best_idx, best_valobj = self._get_best_expert(obs)
        
        if self._uncertainty == "std":
            explore=best_valobj.std < self.sigma
            return explore, best_idx, best_valobj, best_valobj.std
        elif self._uncertainty == "max_gap":
            explore=best_valobj.max_gap < self.sigma
            return explore, best_idx, best_valobj, best_valobj.max_gap
        else:
            raise ValueError(f"Invalid uncertainty type: {self._uncertainty}")


class ActivePolicySelector:
    """Select the best expert to rollout based on the predicted value at the switching state."""
    def __init__(self, value_fns: List[ValueEnsemble]) -> None:
        self.value_fns = value_fns

    def _get_best_expert(self, obs):
        # Find the value function whose upper bound is the best
        sorted_pairs = sorted([(idx, vfn.forward_stats(obs)) for idx, vfn in enumerate(self.value_fns)], key=lambda x: x[1].upper)
        best_idx, best_valobj = sorted_pairs[-1]

        return best_idx, best_valobj

    @torch.no_grad()
    def select(self, obs):
        obs = to_torch(obs).unsqueeze(0)
        best_idx, best_valobj = self._get_best_expert(obs)

        return best_idx, best_valobj


class MambaAgent(Agent):
    """Mamba learner agent.

    This is a simple agent with these properties:
    - Its value function vfn_aggr is max over expert value functions (f^{max})
    - update() method only updates its policy \\pi, using (importance weighted) policy gradient
    """

    def __init__(self, pi: nn.Module, vfn_aggr: nn.Module, optimizer,
                 obs_normalizer: Callable | None = None, max_grad_norm: None | float = None,
                 standardize_advantages: bool = True, gamma: float = 1., lambd: float = 0.9) -> None:
        super().__init__(pi, vfn_aggr, optimizer, obs_normalizer)

        self.max_grad_norm = max_grad_norm
        self.standardize_advantages = standardize_advantages

        self.gamma = gamma
        self.lambd = lambd

        self.vfn_aggr = self.vfn

    def update(self, transitions: Sequence[dict], num_epochs: int = 10, batch_size: int = 128):
        """Only update the actor (not the critic) with `transitions` that have `adv`, `log_prob` attached."""
        if self.standardize_advantages:
            advs = [tr['adv'] for tr in transitions]
            std_advs, mean_advs = torch.std_mean(to_torch(advs), unbiased=False)

        sample_itr = yield_batch_infinitely(transitions, batch_size)
        loss_dict = defaultdict(list)

        num_updates = max(len(transitions) // batch_size, 1) * num_epochs
        for _ in range(num_updates):
            batch = next(sample_itr)

            states = batch['state'].type(torch.float32)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            actions = batch['action'].type(torch.float32)
            distribs = self.pi(states)

            advs = batch['adv'].type(torch.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = batch['log_prob'].type(torch.float32)

            self.optimizer.zero_grad()
            loss = calc_loss_actor(self.pi, states, actions, advs, log_probs_old)

            if loss is not None:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Append to loss dict
                loss_dict['actor'].append(loss.item())
                loss_dict['entropy'].append(torch.mean(distribs.entropy()).item())

            self.n_updates += 1

        loss_info = {
            f'loss/{key}': np.asarray(arr).mean() for key, arr in loss_dict.items()
        }

        return loss_info
