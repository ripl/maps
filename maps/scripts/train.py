#!/usr/bin/env python3

import functools
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import wandb
from maps import logger
from maps.agents.base import Agent
from maps.agents.expert import ExpertAgent
from maps.agents.mamba import (ActivePolicySelector, ActiveStateExplorer,
                                MaxValueFn, ValueEnsemble)
from maps.agents.ppo import PPOAgent, update_critic_ensemble
from maps.evaluation import Evaluator
from maps.helpers import (flatten, limit_num_transitions, set_random_seed,
                           to_torch)
from maps.helpers.env import (PolMvAvg, roll_in_and_out_maps,
                               roll_in_and_out_mamba, rollout,
                               rollout_single_ep)
from maps.helpers.factory import Factory
from maps.helpers.initializers import ortho_init
from maps.nn.empirical_normalization import EmpiricalNormalization
from maps.policies import (GaussianHeadWithStateIndependentCovariance,
                            SoftmaxCategoricalHead)
from maps.scripts.default_args import Args
from maps.value_estimations import (
    _attach_advantage_and_value_target_to_episode,
    _attach_log_prob_to_episodes,
    _attach_mean_return_and_value_target_to_episode,
    _attach_return_and_value_target_to_episode, _attach_value_to_episodes)
from torch import nn


class SwitchingTimeSampler:
    """Sample a switching time, optionally with moving average
    """
    def __init__(self, time_limit, use_moving_ave=True) -> None:
        print('use_moving_ave:', use_moving_ave)
        self.time_limit = time_limit
        self.use_moving_ave = use_moving_ave
        self.pol_mv_avg = PolMvAvg(None)

    def update_pma(self, val):
        """Feed average learner episode lengths."""
        self.pol_mv_avg.update(val)

    def sample(self):
        if self.use_moving_ave:
            t_switch, scale = self._geometric_t(self.pol_mv_avg.val)
        else:
            t_switch = random.randint(0, self.time_limit - 1)

        return t_switch

    def _geometric_t(self, mean, gamma=1.0):
        """Copied from microsoft/mamba."""
        prob = 1 / mean
        t_switch = np.random.geometric(prob)  # starts from 1
        if t_switch > self.time_limit - 1:
            t_switch = self.time_limit - 1
            p = (1 - prob) ** t_switch  # tail probability
        else:
            p = (1 - prob) ** (t_switch - 1) * prob
        prob, scale = self._compute_prob_and_scale(t_switch, gamma)

        return t_switch, prob / p

    def _compute_prob_and_scale(self, t, gamma):
        """Treat the weighting in a problem as probability. Compute the
        probability for a time step and the sum of the weights.

        For the objective below,
            \sum_{t=0}^{T-1} \gamma^t c_t
        where T is finite and \gamma in [0,1], or T is infinite and gamma<1.
        It computes
            scale = \sum_{t=0}^{T-1} \gamma^t
            prob = \gamma^t / scale

        Copied from microsoft/mamba
        """
        assert t <= self.time_limit - 1
        if self.time_limit < float("Inf"):
            p0 = gamma ** np.arange(self.time_limit)
            sump0 = np.sum(p0)
            prob = p0[t] / sump0
        else:
            raise NotImplementedError('I have no idea what to do')
        return prob, sump0


def make_env(test=False, default_seed=0):
    from maps.helpers import env
    env_name = Args.env_name
    seed = default_seed if not test else 42 - default_seed
    extra_kwargs = {'task_kwargs': {'random': seed}}
    return env.make_env(env_name, seed=seed, **extra_kwargs)


def should_flush(episodes, curr_episode, buffer_size):
    """Computes the total number of transitions in the `episodes` and `curr_episode`,
    and returns True if it goes over `buffer_size`.
    """
    dataset_size = (
        sum(len(episode) for episode in episodes)
        + len(curr_episode)
    )

    return dataset_size >= buffer_size


def collect_expert_episodes(env, experts: List[Agent], max_episode_len: int, num_rollouts: int,
                            learner_gamma: float, learner_lmd: float):
    """Roll out experts to collect transitions."""

    expert2episodes = [[] for _ in experts]
    for _ in range(num_rollouts):
        for expert_idx, expert in enumerate(experts):
            episode = rollout_single_ep(env, functools.partial(expert.act, mode=Args.deterministic_experts), max_episode_len)
            expert2episodes[expert_idx].append(episode)

    return expert2episodes


def update_expert_vfn(experts, expert_rollouts, num_epochs, batch_size: int, num_val_iterations: int,
                      learner_gamma: float, learner_lmd: float) -> List[List[dict]]:
    """Update expert value functions based on `expert_rollouts`."""
    logs_per_expert = [[] for _ in experts]

    if num_epochs == 0 or num_val_iterations == 0:
        return logs_per_expert

    for expert_idx, expert in enumerate(experts):
        expert_k_transitions = flatten(expert_rollouts[expert_idx])

        if len(expert_k_transitions) < 2:
            print(f'roll out by the expert {expert_idx} was too short!')
            print('len(expert_k_transitions)', len(expert_k_transitions))
            continue

        itr = 0
        for i in range(num_val_iterations):
            _attach_value_to_episodes(experts[expert_idx].vfn, expert_k_transitions, obs_normalizer=experts[expert_idx].obs_normalizer)

            # Recompute the target value and attach them if necessary
            for episode in expert_rollouts[expert_idx]:
                _attach_advantage_and_value_target_to_episode(episode, learner_gamma, learner_lmd)

            # NOTE: num_updates may change the behavior quite a lot.
            _, loss_critic_history = update_critic_ensemble(expert, expert_k_transitions, num_epochs=max(1, num_epochs // num_val_iterations), batch_size=batch_size, std_from_means=True)
            for loss in loss_critic_history:
                logs_per_expert[expert_idx].append({
                    f'loss-critic-{expert_idx}': loss,
                    f'num-transitions-{expert_idx}': len(expert_k_transitions),
                    'vi-step': itr,
                })
                itr += 1
    return logs_per_expert


def roll_in_and_out(env, learner: Agent, experts: Sequence[Agent], swtime_sampler: SwitchingTimeSampler,
                    num_rollouts: int, gamma: float, lmd: float, max_episode_len: int, return_wandblogs: bool = True,
                    ase_sigma: Optional[float] = None, expert2episodes: Optional[Callable] = None,
                    switching_state_callback: Optional[Callable] = None):
    """Roll in learner policy up to a switching step, select an expert to roll out and roll it out till the end of the episode.
    """
    assert Args.algorithm != 'pg-gae'

    all_episodes = []
    switch_times = []
    ro_expert_inds = []

    log_expidx = []
    log_switch_time = []
    log_expert_traj_len = []
    log_switch_valmeans = []
    log_switch_valstds = []
    log_uncertainty_max = []
    log_uncertainty_min = []
    log_ase_sigma = []

    for _ in range(num_rollouts):
        # Roll-in and out with each "switching" policy and oracle selection policy
        if Args.algorithm == 'maps-se':
            switching_time = swtime_sampler.sample()
            episode, switching_time, expert_idx, values_at_switching_state, uncertainty_max, uncertainty_min = roll_in_and_out_maps(
                env,
                learner.act,
                [functools.partial(expert.act, mode=Args.deterministic_experts) for expert in experts],
                active_state_explorer=ActiveStateExplorer(value_fns=[expert.vfn for expert in experts], sigma=ase_sigma, uncertainty="std"),
                active_policy_selector=ActivePolicySelector(value_fns=[expert.vfn for expert in experts]),
                max_episode_len=max_episode_len,
                switch_time=None,
                # switching_state_callback=switching_state_callback
            )
        elif Args.algorithm == 'maps':
            switching_time = swtime_sampler.sample()
            episode, switching_time, expert_idx, values_at_switching_state, uncertainty_max, uncertainty_min = roll_in_and_out_maps(
                env,
                learner.act,
                [functools.partial(expert.act, mode=Args.deterministic_experts) for expert in experts],
                active_state_explorer=None,
                active_policy_selector=ActivePolicySelector(value_fns=[expert.vfn for expert in experts]),
                max_episode_len=max_episode_len,
                switch_time=switching_time,
                # switching_state_callback=switching_state_callback
            )
            uncertainty_min = None
            uncertainty_max = None
        elif Args.algorithm == 'mamba':
            switching_time = swtime_sampler.sample()
            expert_idx = random.randint(0, len(experts) - 1)
            episode, values_at_switching_state = roll_in_and_out_mamba(env, learner.act, experts[expert_idx].act, experts[expert_idx].vfn, switching_time, max_episode_len)
            uncertainty_min = None
            uncertainty_max = None
        else:
            raise ValueError(f'Unknown algorithm: {Args.algorithm}')

        all_episodes.append(episode)
        switch_times.append(switching_time)
        ro_expert_inds.append(expert_idx)

        log_switch_time.append(switching_time)
        log_expert_traj_len.append(len(episode) - switching_time)

        if values_at_switching_state is not None:
            log_switch_valmeans.append(values_at_switching_state.mean.item())
            log_switch_valstds.append(values_at_switching_state.std.item())

        if uncertainty_min is not None:
            if isinstance(uncertainty_min, torch.Tensor):
                log_uncertainty_min.append(uncertainty_min.cpu().detach().numpy())
                log_uncertainty_max.append(uncertainty_max.cpu().detach().numpy())
            else:
                log_uncertainty_min.append(uncertainty_min)
                log_uncertainty_max.append(uncertainty_max)

            if Args.use_ase_sigma_ratio:
                ase_sigma = np.min(log_uncertainty_min) + Args.ase_sigma_ratio * (
                    np.max(log_uncertainty_max) - np.min(log_uncertainty_min)
                )
                log_ase_sigma.append(ase_sigma)

        # Just for logging
        if switching_time < len(episode):  # NOTE: 0 <= switching_time <= len(episode)
            # Expert policy is rolled out at least one step.
            log_expidx.append(expert_idx)

    if return_wandblogs:
        print('switch times', log_switch_time)
        print('selected_expert', log_expidx)
        wandb_logs = ({
            'riro/learner-swtchtime-hist': wandb.Histogram(log_switch_time),
            'riro/learner-swtchtime-mean': np.mean(log_switch_time),
            'riro/learner-swtchtime-min': min(log_switch_time),
            'riro/learner-swtchtime-max': max(log_switch_time),
            'riro/expert_traj_len-hist': wandb.Histogram(log_expert_traj_len),
            'riro/expert_traj_len-min': min(log_expert_traj_len),
            'riro/expert_traj_len-mean': np.mean(log_expert_traj_len),
            'riro/expert_traj_len-max': max(log_expert_traj_len),
            'riro/selected_expert': wandb.Histogram(log_expidx),
            **{f'riro/selected_expert-{expidx}': (np.asarray(log_expidx) == expidx).mean() for expidx in range(len(experts))},
        })

        if len(log_switch_valmeans) > 0:
            wandb_logs = {
                'riro/selected_expert_val_mean': np.mean(log_switch_valmeans),
                'riro/selected_expert_val_std': np.mean(log_switch_valstds),
                **wandb_logs
            }

        if len(log_ase_sigma) > 0:
            wandb_logs = {
                'riro/ase_sigma': np.mean(log_ase_sigma),
                **wandb_logs
            }

        # Track min max uncertainty
        if len(log_uncertainty_max) > 0 and len(log_uncertainty_min) > 0:

            wandb_logs = {
                'riro/uncertianty_max': np.max(log_uncertainty_max),
                'riro/uncertianty_min': np.min(log_uncertainty_min),
                **wandb_logs
            }

        return all_episodes, switch_times, ro_expert_inds, wandb_logs

    return all_episodes, switch_times, ro_expert_inds


def train_lops(make_env: Callable, evaluator: Evaluator, learner: Agent, experts: List[Agent], swtime_sampler: SwitchingTimeSampler, num_train_steps: int, max_episode_len = 1000, eval_every: int = 1):
    env = make_env()

    # stddev_baseline = None
    if Args.algorithm != 'pg-gae':
        # Rollout 10 episodes just to get the initial average episode lengths
        ep_lengths = []
        for _ in range(10):
            episode = rollout_single_ep(env, functools.partial(learner.act, mode=Args.deterministic_experts), max_episode_len)
            ep_lengths.append(len(episode))
        swtime_sampler.update_pma(np.mean(ep_lengths))  # Needs to set the expected episode length

        # Let's evaluate the experts to benchmark them!
        mean2stddev = []
        for expert_idx, expert in enumerate(experts):
            logs = evaluator.evaluate(expert, num_eval_episodes=1)
            mean2stddev.append((logs['eval/returns_mean'], logs['eval/returns_std']))
            logs = {f'prep/expert-{expert.name}-{key}': val for key, val in logs.items() if not key.startswith('_')}
            wandb.log({'step': 0, **logs})

        # NOTE: logs.keys == ("returns_std", "returns_mean")
        # TODO: Get the stddev of the best expert policy
        # stddev_baseline = sorted(mean2stddev, reverse=True)[0][1]
        # print('mean2stddev', mean2stddev)
        # print('stddev baseline', stddev_baseline)
        # wandb.log({'step': 0, 'prep/stddev_baseline': stddev_baseline})

    # Rollout the experts and pretrain value functions
    if Args.algorithm == 'pg-gae':
        # For pg-gae, forget about it
        expert2episodes = []
    else:
        # For each expert k, collect data D^k by rolling out pi^k
        logger.info('Collecting expert episodes...')
        expert2episodes = collect_expert_episodes(env, experts, max_episode_len,
                                                  num_rollouts=Args.pret_num_rollouts,
                                                  learner_gamma=learner.gamma, learner_lmd=learner.lambd)
        logger.info('Collecting expert episodes...done')

        # Restrict the size of expert_episodes (simulating FIFO buffer)!!
        for expert_idx, expert_eps in enumerate(expert2episodes):
            expert2episodes[expert_idx] = limit_num_transitions(expert_eps, max_transitions=Args.expert_buffer_size)

        # Expose the transitions to expert and obs normalizer
        # _transitions = flatten([flatten(episodes) for expert_idx, episodes in enumerate(expert2episodes)])
        # for expert in experts:
        #     _states = to_torch([tr['state'] for tr in _transitions])
        #     # expert.obs_normalizer.experience(_states)

        # Pretrain value functions
        # Update value function V^k from samples in D^k
        # TODO: Change this to use expert_vfns rather than experts
        logger.info('Updating exeperts value functions...')
        log_list = update_expert_vfn(experts, expert2episodes, num_epochs=Args.pret_num_epochs, batch_size=Args.batch_size,
                                     num_val_iterations=Args.pret_num_val_iterations, learner_gamma=learner.gamma,
                                     learner_lmd=learner.lambd)
        logger.info('Updating exeperts value functions...done')  # NOTE ^ This attaches v_teacher key to each transition

        for expert_idx, logs in enumerate(log_list):
            for log in logs:
                wandb.log({f'pretrain/expert-{key}' if key != 'vi-step' else key: val for key, val in log.items()})

    # Main training loop
    for itr in range(num_train_steps):
        logger.info(f'LOPS training loop {itr + 1} / {num_train_steps}')
        learner_ep_lens = []
        learner_episodes = []

        # Evaluate the models
        if itr % eval_every == 0:
            logs = evaluator.evaluate(learner, num_eval_episodes=Args.num_eval_episodes, update_best=True)
            wandb.log({'step': itr, **logs})

        # Roll-in and out, and store the episodes to expert_rollouts
        if Args.algorithm != 'pg-gae':
            ase_sigma = Args.ase_sigma

            all_episodes, switching_times, ro_expert_inds, wandb_logs = roll_in_and_out(
                env, learner, experts, swtime_sampler, Args.num_rollouts // 2, learner.gamma,
                learner.lambd, max_episode_len, return_wandblogs=True, ase_sigma=ase_sigma, expert2episodes=expert2episodes
            )
            wandb.log({'step': itr, **wandb_logs})

            for _episode, _sw_time, _ro_exp_idx in zip(all_episodes, switching_times, ro_expert_inds):

                # Merge newly obtained expert_rollouts to the current expert_rollouts
                if len(_episode[_sw_time:]) > 0:
                    expert2episodes[_ro_exp_idx].append(_episode[_sw_time:])

                    # Expose new transitions to expert and state-predictor's obs_normalizers
                    _states = to_torch([tr['state'] for tr in _episode[_sw_time:]])
                    if experts[_ro_exp_idx].obs_normalizer is not None:
                        experts[_ro_exp_idx].obs_normalizer.experience(_states)

                # Merge newly obtained leaner_rollouts to the current learner_rollouts
                if len(_episode[:_sw_time]) > 0:  # NOTE: _sw_time may be bigger than len(_episode) itself!
                    if Args.use_riro_for_learner_pi != 'none':
                        # Do not include the expert rollout in learner_episodes
                        # NOTE: learner's obs_normalizer will be updated right before its policy update
                        learner_episodes.append(_episode[:_sw_time])

            # Limit the size of expert rollouts (simulating FIFO buffer)
            for expert_idx, expert_eps in enumerate(expert2episodes):
                expert2episodes[expert_idx] = limit_num_transitions(expert_eps, max_transitions=Args.expert_buffer_size)

            # Update value model V^k from samples in D^k
            log_list = update_expert_vfn(experts, expert2episodes, num_epochs=Args.num_epochs,
                                         batch_size=Args.batch_size, num_val_iterations=1,
                                         learner_gamma=learner.gamma, learner_lmd=learner.lambd)

            for expert_idx, logs in enumerate(log_list):
                keys = logs[0].keys()
                _logs = {f'train-expert/{key}': np.mean([log[key] for log in logs]) for key in keys if key != 'vi-step'}
                wandb.log({**_logs, 'step': itr})

        # Rollout entirely with the learner policy to collect data D'^n
        _episodes, curr_ep = rollout(env, learner.act, max_episode_len, break_cond=functools.partial(should_flush, buffer_size=Args.learner_buffer_size))
        if len(curr_ep) > 0:  # Merge all episodes
            _episodes.append(curr_ep)
        for ep in _episodes:
            learner_episodes.append(ep)
        learner.obs_normalizer.experience(to_torch([tr['state'] for tr in flatten(learner_episodes)]))  # Critical: update obs_normalizer right before learner.update

        # Attach predicted values and log_pi to the learner rollouts to prepare for learner update
        if Args.algorithm == 'pg-gae':
            _learner_transitions = flatten(learner_episodes)
            _attach_value_to_episodes(learner.vfn, _learner_transitions, obs_normalizer=learner.obs_normalizer)
            _attach_log_prob_to_episodes(learner.pi, _learner_transitions, obs_normalizer=learner.obs_normalizer)
        else:
            # Use obs_normalizers in MaxValueFn
            _learner_transitions = flatten(learner_episodes)
            _attach_value_to_episodes(functools.partial(learner.vfn, normalize_input=True), _learner_transitions, obs_normalizer=None)
            _attach_log_prob_to_episodes(learner.pi, _learner_transitions, obs_normalizer=learner.obs_normalizer)

            # Attach experts' logpi for (partial) expert trajectories
            if Args.use_riro_for_learner_pi == 'all':
                assert len(experts) == len(expert2episodes)
                for expert, expert_eps in zip(experts, expert2episodes):
                    expert_eps = deepcopy(expert_eps)  # NOTE: We need deepcopy to make sure not to attach logp to the original expert transitions
                    expert_transitions = flatten(expert_eps)

                    # NOTE When Args.deterministic_experts is False (i.e., expert transitions are collected in a stochastic manner),
                    # somehow the log probability becomes excessively small (~ -1000) and this causes probability ratio in PPO loss to blow up,
                    # making the loss infinity immediately.
                    _attach_log_prob_to_episodes(expert.pi, expert_transitions, obs_normalizer=expert.obs_normalizer)
                    # print(f'exp: {expert} transitions', len(expert_transitions))
                    # log_probs = np.asarray([tr['log_prob'] for tr in expert_transitions])
                    # print(f'exp transitions log_prob max {log_probs.max()}\tmin {log_probs.min()}\tisnan any {np.isnan(log_probs).any()}')

                    # Merge expert_eps to learner_episodes
                    learner_episodes.extend(expert_eps)

        # Compute advantage (GAE-lambda)
        for episode in learner_episodes:
            _attach_advantage_and_value_target_to_episode(episode, learner.gamma, learner.lambd)
            learner_ep_lens.append(len(episode))

        # Update the learner policy using the learner-rollout trajectory that is bootstrapped with expert value functions.
        # This also updates the critic if algorithm is "pg-gae"
        learner_transitions = flatten(learner_episodes)
        logger.info('Updating the laerner policy...')
        loss_info = learner.update(learner_transitions, num_epochs=Args.num_epochs, batch_size=Args.batch_size)
        logs = {
            f'learner/{key.replace("/", "-")}': val for key, val in loss_info.items()
        }

        wandb.log({
            'step': itr,
            'train-learner/num_transitions': len(learner_transitions),
            'train-learner/episode_lens': wandb.Histogram(learner_ep_lens),
            'train-learner/episode_lens_mean': np.mean(learner_ep_lens),
            **logs,
        })

        # Update polynomial moving average based on learner episode lengths
        swtime_sampler.update_pma(np.mean(learner_ep_lens))


def main():
    import gym
    from maps.agents.mamba import MambaAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(Args.seed)

    test_env = make_env()
    assert isinstance(test_env.action_space, gym.spaces.Box), "This script only works with continuous action space"

    # Set up the policy head
    state_dim = test_env.observation_space.low.size
    act_dim = test_env.action_space.low.size
    policy_head = GaussianHeadWithStateIndependentCovariance(
        action_size=act_dim,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    )

    # Set up the learner policy
    learner_pi = Factory.create_pi(state_dim, act_dim, policy_head=policy_head)
    obs_normalizer = EmpiricalNormalization(state_dim, clip_threshold=5)
    obs_normalizer.to(device)

    # Load experts
    experts = [ExpertAgent(test_env, model_dir, policy=policy) for policy, model_dir in Args.experts_info]

    # Define Value Ensemble and Initialize weights
    make_vfn = lambda: Factory.create_vfn(state_dim, mean_and_var=True, initializer=None, activation=nn.ReLU)
    for expert in experts:
        value_ensemble = ValueEnsemble(make_vfn, Args.num_expert_vfns, std_from_means=Args.std_from_means)
        for _vfn in value_ensemble.vfns:
            ortho_init(_vfn[0], gain=Args.expert_vfn_gain, zeros_bias=False)
            ortho_init(_vfn[2], gain=Args.expert_vfn_gain, zeros_bias=False)
            ortho_init(_vfn[4], gain=Args.expert_vfn_gain, zeros_bias=False)

        # Set value function and optimizer to each expert
        expert.vfn = value_ensemble
        expert.optimizer = torch.optim.Adam(expert.vfn.parameters(), lr=1e-3)
        expert.to(device)

    max_vfn = MaxValueFn([expert.vfn for expert in experts])

    if Args.algorithm == 'pg-gae':
        _vfn = Factory.create_vfn(state_dim)
        optimizer = torch.optim.Adam(list(learner_pi.parameters()) + list(_vfn.parameters()), lr=1e-3, betas=(0.9, 0.99))
        learner = PPOAgent(learner_pi, _vfn, optimizer, obs_normalizer, gamma=Args.gamma, lambd=Args.lmd)
        _vfn.to(device)
    else:
        optimizer = torch.optim.Adam(learner_pi.parameters(), lr=1e-3, betas=(0.9, 0.99))
        learner = MambaAgent(learner_pi, max_vfn, optimizer, obs_normalizer, gamma=Args.gamma, lambd=Args.lmd,
                             max_grad_norm=Args.max_grad_norm)
    learner_pi.to(device)
    learner.to(device)

    swtime_sampler = SwitchingTimeSampler(time_limit=Args.max_episode_len, use_moving_ave=False)
    evaluator = Evaluator(make_env, max_episode_len=Args.max_episode_len)

    train_lops(make_env, evaluator, learner, experts, swtime_sampler, num_train_steps=Args.num_train_steps, max_episode_len=Args.max_episode_len)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", help="sweep file")
    parser.add_argument("-l", "--line-number", type=int, help="sweep file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    from params_proto.hyper import Sweep
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    # Update the parameters
    Args._update(kwargs)

    if Args.algorithm == 'pg-gae':
        # Add (maximum) number of transitions the model would experience if roll-in-and-out is performed
        Args.learner_buffer_size += Args.max_episode_len * (Args.num_rollouts // 2)

    sweep_basename = Path(args.sweep_file).stem

    # Wandb setup
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=f'maps-{sweep_basename}',
        config=vars(Args),
    )

    main()
