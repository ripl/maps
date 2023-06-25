#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path

import torch
import wandb
from maps.policies import GaussianHeadWithStateIndependentCovariance
from maps.scripts.default_args import Args
from maps.scripts.train import Evaluator, get_expert


def main(load_expert_steps, evaluator):

    test_env = make_env()
    state_dim = test_env.observation_space.low.size
    act_dim = test_env.action_space.low.size
    policy_head = GaussianHeadWithStateIndependentCovariance(
        action_size=act_dim,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    )
    experts = []
    for idx in load_expert_steps:
        expert = get_expert(state_dim, act_dim, deepcopy(policy_head), Path(Args.experts_dir) / test_env.unwrapped.spec.id.lower() / f'step_{idx:06d}.pt', obs_normalizer=None)
        experts.append(expert)


    ## Let's evaluate the experts first!
    for expert_idx, expert in enumerate(experts):
        logs = evaluator.evaluate(expert, num_eval_episodes=32)
        logs = {f'prep/expert-{load_expert_steps[expert_idx]}-{key}': val for key, val in logs.items() if not key.startswith('_')}
        wandb.log({'step': 0, **logs})


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("envname", type=str, help="env name")
    parser.add_argument("--load-expert-steps", nargs='+', type=int, help="list of expert steps")
    args = parser.parse_args()
    max_episode_len = 300

    def _make_env(env_name='DartCartPole-v1', test=False, default_seed=0):
        """
        Environments we want to try:
        - dmc:Cheetah-run-v1
        - dmc:Ant-walk-v1
        - DartCartPole-v1
        - DardDoubleInvertedPendulum-v1  <-- not tested yet
        """
        from maps.helpers import env
        seed = default_seed if not test else 42 - default_seed
        if env_name.startswith('dmc'):
            extra_kwargs = {'task_kwargs': {'random': seed}}
        else:
            extra_kwargs = {}
        return env.make_env(env_name, seed=seed, **extra_kwargs)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project='maps-eval-experts',
        config=vars(Args),
    )

    make_env = lambda *_args, **_kwargs: _make_env(args.envname, *_args, **_kwargs)  # TEMP
    evaluator = Evaluator(make_env, max_episode_len=max_episode_len)
    main(args.load_expert_steps, evaluator)
