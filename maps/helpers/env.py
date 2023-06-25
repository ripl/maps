#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable, Optional, Sequence

import gym
import numpy as np
import pfrl
import torch
from maps.helpers import to_torch
from torch import nn


def make_env(envid, seed: int, test=False, **kwargs) -> gym.Env:
    env = gym.make(envid, **kwargs)
    env_seed = 2**32 - 1 - seed if test else seed
    env.seed(env_seed)

    # NOTE: random seed in env.action_space is separate and cannote be set by env.seed()
    env.action_space.seed(env_seed)

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)

    # Normalize action space to [-1, 1]^n
    env = pfrl.wrappers.NormalizeActionSpace(env)

    return env


def make_batch_env(envid: str, num_envs: int, test=False):
    import functools

    import pfrl
    return pfrl.envs.MultiprocessVectorEnv(
        [
            functools.partial(make_env, envid=envid, seed=idx, test=test)
            for idx, env in enumerate(range(num_envs))
        ]
    )


def rollout(env: gym.Env, policy: Callable, max_episode_len: int, break_cond: Optional[Callable] = None,
            reset_fn: Optional[Callable] = None, save_sim_state: bool = False, save_video: bool = False):
    episodes = []
    curr_episode = []

    episode_r = 0
    episode_len = 0
    episode_idx = 0

    env_reset = env.reset if reset_fn is None else lambda: reset_fn(env)
    obs = env_reset()
    reset = False

    while True:
        # a_t
        action = policy(obs)
        next_obs, r, done, info = env.step(action)

        episode_r += r
        episode_len += 1

        # Compute mask for done and reset
        if max_episode_len is not None:
            reset = reset or episode_len == max_episode_len
        reset = reset or info.get('needs_reset', False)

        # Make mask. 0 if done/reset, 1 if pass
        end = reset or done

        transition = {
            "state": obs,
            "action": action,
            "reward": r,
            "next_state": next_obs,
            "nonterminal": 0.0 if done else 1.0,
        }
        if save_sim_state:
            transition['sim_state'] = env.unwrapped.env.physics.get_state()

        if save_video:
            transition['frame'] = env.render(mode='rgb_array', width=112, height=112)

        curr_episode.append(transition)

        # Critical
        obs = next_obs

        # Start new episodes if needed
        if end:
            episodes.append(curr_episode)
            curr_episode = []
            episode_r = 0
            episode_len = 0
            episode_idx += 1
            reset = False
            obs = env_reset()  # This only resets necessary envs

            if break_cond is None:
                break

        if break_cond is not None and break_cond(episodes, curr_episode):
            break

    return episodes, curr_episode


def rollout_single_ep(env, policy: Callable, max_episode_len: int, **kwargs):
    """Roll out a single episode and return the episode"""
    ep, _ = rollout(env, policy, max_episode_len, **kwargs)
    return ep[0]


def rollout_from_state(env, policy: Callable, sim_state: np.ndarray, init_obs: np.ndarray, max_episode_len: int, **kwargs):
    """Rollout the given policy starting at sim_state.
    """
    def env_reset(env):
        # Restore state
        env.reset()
        env.unwrapped.env.physics.set_state(sim_state)
        return init_obs

    ep = rollout_single_ep(env, policy, max_episode_len, reset_fn=env_reset, **kwargs)
    return ep


@torch.no_grad()
def roll_in_and_out_maps(
    env,
    learner_policy: Callable,
    expert_policies: Sequence[Callable],
    active_state_explorer,
    active_policy_selector,
    max_episode_len,
    switch_time=None,
    switching_state_callback: Optional[Callable] = None,
):
    """Roll-in with learner policy up to t = switching_time, and then roll-out expert till the end."""
    step, episode_r = 0, 0

    obs = env.reset()
    reset = False

    learner_traj = []
    expert_traj = []
    end = False
    uncertainty_val_arr = []

    # Explore with learner policy
    while not end:
        if active_state_explorer is None and switch_time is not None:
            if step < switch_time:
                should_explore = True
            else:
                should_explore = False
                expert_idx, valobj = active_policy_selector.select(obs)

        elif active_state_explorer is not None and switch_time is None:
            should_explore, expert_idx, valobj, uncertainty_val = active_state_explorer.should_explore(obs)
            uncertainty_val_arr.append(uncertainty_val)
        else:
            print("roll_in_and_out_maps setting error")
            exit()

        if not should_explore:
            break

        action = learner_policy(obs)

        next_obs, r, done, info = env.step(action)

        episode_r += r
        step += 1

        # Compute mask for done and reset
        if max_episode_len is not None:
            reset = reset or step == max_episode_len
        reset = reset or info.get('needs_reset', False)

        # Make mask. 0 if done/reset, 1 if pass
        end = reset or done

        transition = {
            "state": obs,
            "action": action,
            "reward": r,
            "next_state": next_obs,
            "nonterminal": 0.0 if done else 1.0,
        }
        learner_traj.append(transition)

        # Critical
        obs = next_obs

    if not should_explore:
        expert_policy = expert_policies[expert_idx]
        values_at_switching_state = valobj
        if len(uncertainty_val_arr) > 0:
            uncertainty_val_max = max(uncertainty_val_arr)
            uncertainty_val_min = min(uncertainty_val_arr)
        else:
            uncertainty_val_max = None
            uncertainty_val_min = None
    else:
        expert_idx = None
        expert_policy = None
        values_at_switching_state = None
        uncertainty_val_max = None
        uncertainty_val_min = None

    # Roll out with the expert
    while not end:
        # a_t
        action = expert_policy(obs)
        next_obs, r, done, info = env.step(action)

        episode_r += r
        step += 1

        # Compute mask for done and reset
        if max_episode_len is not None:
            reset = reset or step == max_episode_len
        reset = reset or info.get('needs_reset', False)

        # Make mask. 0 if done/reset, 1 if pass
        end = reset or done

        transition = {
            "state": obs,
            "action": action,
            "reward": r,
            "next_state": next_obs,
            "nonterminal": 0.0 if done else 1.0,
        }
        expert_traj.append(transition)

        # Critical
        obs = next_obs

    whole_traj = learner_traj + expert_traj
    return whole_traj, len(learner_traj), expert_idx, values_at_switching_state, uncertainty_val_max, uncertainty_val_min


@torch.no_grad()
def roll_in_and_out_mamba(env, learner_policy: Callable, expert_policy: Callable, expert_vfn: nn.Module, switching_time, max_episode_len):
    """Roll-in with learner policy up to t = switching_time, and then roll-out expert till the end."""
    assert 0 <= switching_time < max_episode_len
    episode = []
    step, episode_r = 0, 0

    obs = env.reset()
    reset = False
    end = False
    values_at_switching_state = None
    while not end:
        # a_t
        if step < switching_time:
            action = learner_policy(obs)
            if step == switching_time - 1:
                values_at_switching_state = expert_vfn.forward_stats(to_torch(obs).unsqueeze(0))
        else:
            action = expert_policy(obs)
        next_obs, r, done, info = env.step(action)

        episode_r += r
        step += 1

        # Compute mask for done and reset
        if max_episode_len is not None:
            reset = reset or step == max_episode_len
        reset = reset or info.get('needs_reset', False)

        # Make mask. 0 if done/reset, 1 if pass
        end = reset or done

        transition = {
            "state": obs,
            "action": action,
            "reward": r,
            "next_state": next_obs,
            "nonterminal": 0.0 if done else 1.0,
        }
        episode.append(transition)

        # Critical
        obs = next_obs

    return episode, values_at_switching_state


class PolMvAvg:
    """An estimator based on polynomially weighted moving average.

    The estimate after N calls is computed as
        val = \sum_{n=1}^N n^power x_n / nor_N
    where nor_N is equal to \sum_{n=1}^N n^power, and power is a parameter.

    Copied from microsoft/mamba
    """

    def __init__(self, val, power=0, weight=0.0):
        self._val = val * weight if val is not None else 0.0
        self._nor = weight
        self.power = power
        self._itr = 1

    def update(self, val, weight=1.0):
        self._val = self.mvavg(self._val, val * weight, self.power)
        self._nor = self.mvavg(self._nor, weight, self.power)
        self._itr += 1

    def mvavg(self, old, new, power):
        return old + new * self._itr ** power

    @property
    def val(self):
        return self._val / np.maximum(1e-8, self._nor)
