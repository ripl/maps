#!/usr/bin/env python3

from pathlib import Path
from .base import Agent
from maps import logger
from maps.scripts.pretraining.utils import get_sac_agent, get_ppo_agent
import numpy as np
from torch import nn

class PFRLAgentAdapter(nn.Module):
    """
    Takes PFRL agent model and convert the interface to lops

    PFRL agent model:
    model  state s  ->  (pi(s, _), v(s))

    LOPS:
           state s  ->  distr
    """
    def __init__(self, agent, policy):
        super().__init__()

        self.policy = policy

        if policy == 'ppo':
            self.model = agent.model
        elif policy == 'sac':
            self.model = agent.policy
        else:
            raise KeyError(f'Unknown policy: {policy}')

    def forward(self, state):
        if self.policy == 'ppo':
            action_distr, value = self.model(state)
        elif self.policy == 'sac':
            assert len(state.shape) >= 2
            action_distr = self.model(state)

        return action_distr


class ExpertAgent:
    """Load from the pretrained agent

    """
    def __init__(self, sample_env, model_dir, policy):
        if policy == 'sac':
            agent = get_sac_agent(sample_env)
        elif policy == 'ppo':
            agent = get_ppo_agent(sample_env)
        else:
            raise KeyError(f'Unknown policy: {policy}')
        self.agent = agent

        logger.info(f'Loading expert weights from {model_dir}...')
        self.agent.load(model_dir)
        logger.info(f'Loading expert weights from {model_dir}...done')

        # For compatibility with Agent class
        self.pi = PFRLAgentAdapter(agent, policy)

        # TODO: Add obs_normalizer and vfn
        self.vfn = None  # To be registered in train.py
        self.optimizer = None  # To be registered in train.py
        self.obs_normalizer = getattr(agent, 'obs_normalizer', None)  # If PPO, reuse the pretrained obs_normalizer

        # Get step from model_dir
        step = int(Path(model_dir).stem.split('_')[0])
        self.name = f'{policy}-{step}'

    def act(self, state, mode=False) -> np.ndarray:
        orig_mode = self.agent.training

        if mode:
            self.agent.training = False
        action = self.agent.act(state)

        self.agent.training = orig_mode
        return action

    def to(self, device):
        self.pi.to(device)
        self.vfn.to(device)
        if hasattr(self, 'obs_normalizer') and self.obs_normalizer is not None:
            self.obs_normalizer.to(device)
