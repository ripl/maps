#!/usr/bin/env python3
from __future__ import annotations

from distutils.version import LooseVersion
from typing import Optional

import cv2
import numpy as np
import pfrl
import torch
import wandb
from pfrl import replay_buffers
from pfrl.experiments.evaluation_hooks import EvaluationHook
from pfrl.nn.lmbda import Lambda
from torch import distributions, nn
import gym


def get_sac_agent(sample_env: gym.Env, policy_output_scale=1., batch_size=256, replay_start_size=10_000):
    """Adopted from https://github.com/pfnet/pfrl/blob/master/examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py"""
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    # print("Observation space:", obs_space)
    # print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=policy_output_scale)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10**6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        replay_start_size=replay_start_size,
        gpu=0,  # Default gpu id to use
        minibatch_size=batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )
    return agent


def get_ppo_agent(sample_env: gym.Env):
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    # print("Observation space:", obs_space)
    # print("Action space:", action_space)

    assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    action_size = action_space.low.size
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = pfrl.agents.PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=0,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )
    return agent


def get_frame(env, episode, step, obs=None, reward=None, reward_sum=None, action: Optional[np.ndarray] = None,
              scale: float | None = None, frame: None | np.ndarray = None, info=None):
    # frame: (h, w, c)
    fontScale = .3

    if frame is None:
        frame = np.ascontiguousarray(env.render(mode='rgb_array'), dtype=np.uint8)
    frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(0, 20), fontFace=3, fontScale=fontScale,
                        color=(0, 255, 0), thickness=1)
    if reward is not None:
        frame = cv2.putText(frame, f'R: {reward:.4f}', org=(0, 60), fontFace=3, fontScale=fontScale, color=(0, 255, 0),
                            thickness=1)
        frame = cv2.putText(frame, f'R-sum: {reward_sum:.4f}', org=(0, 100), fontFace=3, fontScale=fontScale,
                            color=(0, 255, 0), thickness=1)
    # if obs is not None:
    #     x, y, vx, vy, rx, ry, rvx, rvy, *_ = obs
    #     frame = cv2.putText(frame, f'(X, Y): ({x:.1f}, {y:.1f})  (VX, VY): ({vx:.1f}, {vy:.1f})', org=(0, 140), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)
    #     frame = cv2.putText(frame, f'(RX, RY): ({rx:.1f}, {ry:.1f})  (RVX, RVY): ({rvx:.1f}, {rvy:.1f})', org=(0, 180), fontFace=3, fontScale=fontScale, color=(0, 255, 0), thickness=2)

    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, dsize=(width, height))

    return frame


def get_eval_frame(env, episode, step, scale: float | None = None, frame: None | np.ndarray = None, info=None):
    # frame: (h, w, c)
    if frame is None:
        frame = np.ascontiguousarray(env.render(mode='rgb_array'), dtype=np.uint8)

    frame = cv2.putText(frame, f'EP: {episode} STEP: {step}', org=(20, 65), fontFace=3, fontScale=1.3,
                        color=(0, 255, 0), thickness=1)

    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, dsize=(width, height))

    return frame


class WandBLogger(EvaluationHook):
    support_train_agent = True

    # support_train_agent_batch = False
    # support_train_agent_async = False
    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats) -> None:
        """Visualize trajectories for 5 episodes """
        num_eval_episodes = 5
        frames = []
        with agent.eval_mode():
            for ep in range(num_eval_episodes):
                _step = 0
                done = False
                obs = env.reset()
                rew_sum = 0
                # TODO: add episode step in the image
                frames.append(get_frame(env, ep, _step, obs))
                while not done:
                    action = agent.act(obs)
                    obs, rew, done, info = env.step(action)
                    _step += 1
                    rew_sum += rew
                    frames.append(get_frame(env, ep, _step, obs, rew, rew_sum, action=action, info=info))

        wandb.log({
            'eval/mean': eval_stats['mean'],
            'eval/median': eval_stats['median'],
            'eval/ep_length': eval_stats['length_mean'],
            # frames: (t, h, w, c) -> (t, c, h, w)
            'eval/video': wandb.Video(np.asarray(frames).transpose(0, 3, 1, 2), fps=30, format='mp4'),
            'step': step
        })


class WandBLoggerBatch(EvaluationHook):
    support_train_agent = True
    support_train_agent_batch = True
    # support_train_agent_async = False
    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats) -> None:
        """Visualize trajectories for 5 episodes """
        wandb.log({
            'eval/mean': eval_stats['mean'],
            'eval/median': eval_stats['median'],
            'eval/ep_length': eval_stats['length_mean'],
            'step': step
        })
