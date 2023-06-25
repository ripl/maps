#!/usr/bin/env python3
from typing import Callable
import numpy as np
import torch

import maps
from maps import logger
from maps.agents.base import Agent


@torch.no_grad()
def eval_fn(make_env: Callable, agent: Agent, max_episode_len, num_episodes: int = 20, save_video_num_ep: int = 0, verbose: bool = False):
    """
    Args:
        - save_video_num_ep: number of episodes to save the frames
    """
    import wandb
    from maps.helpers.env import rollout_single_ep

    env = make_env(test=True)
    returns = []
    ep_lens = []
    frames = []

    def policy(obs: np.ndarray):
        return agent.act(obs, mode=True)

    if verbose:
        logger.info('Running evaluation...')

    with maps.helpers.evaluating(agent.pi):
        for ep_idx in range(num_episodes):
            ep = rollout_single_ep(env, policy, max_episode_len, save_video=(ep_idx < save_video_num_ep))
            returns.append(sum(transition['reward'] for transition in ep))
            ep_lens.append(len(ep))

            if verbose:
                logger.info(f'eval {ep_idx + 1} / {num_episodes} -- return: {returns[-1]}')

            if 'frame' in ep[0]:
                frames += [transition['frame'] for transition in ep]

    out = {'returns_mean': np.array(returns).mean(),
           'returns_std': np.array(returns).std(),
           'returns': wandb.Histogram(returns),
           'ep_lens_mean': np.array(ep_lens).mean(),
           'ep_lens': wandb.Histogram(ep_lens),
           '_returns': returns}
    if len(frames) > 0:
        out['frames'] = frames
    return out


class Evaluator:
    def __init__(self, make_env, max_episode_len) -> None:
        self.make_env = make_env
        self.max_episode_len = max_episode_len
        self.best_so_far = -np.inf

    def evaluate(self, agent, num_eval_episodes, update_best=False, save_video_num_ep=0):
        stats = eval_fn(self.make_env, agent, self.max_episode_len, num_episodes=num_eval_episodes, save_video_num_ep=save_video_num_ep, verbose=True)
        logs = {f'eval/{key}': val for key, val in stats.items() if not key.startswith('_')}

        if update_best:
            self.best_so_far = max(self.best_so_far, np.mean(stats['_returns']), )
            logs = {'eval/best-so-far': self.best_so_far, **logs}

        return logs
