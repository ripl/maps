"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.

This script is adopted from https://github.com/pfnet/pfrl/blob/master/examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py
"""
import argparse
import functools
import logging
import os
import sys
from pathlib import Path

import gym
import gym.wrappers
import numpy as np
import pfrl
import torch
import wandb
from alops.helpers.env import make_batch_env, make_env
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from torch import distributions, nn

from .utils import WandBLoggerBatch, get_sac_agent, get_ppo_agent

TIME_LIMIT = 1000


def main(args, outdir):
    logging.basicConfig(level=args.log_level)

    # outdir = experiments.prepare_output_dir(args, outdir, argv=sys.argv)
    print("Output files are saved in {}".format(outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    sample_env = make_env(args.env_name, seed=0)
    if args.policy == 'sac':
        agent = get_sac_agent(sample_env)
    elif args.policy == 'ppo':
        agent = get_ppo_agent(sample_env)
    else:
        raise ValueError(f'Unknown policy: {args.policy}')

    experiments.train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_env(args.env_name, num_envs=args.num_envs, test=False),
        eval_env=make_batch_env(args.env_name, num_envs=args.num_envs, test=True),
        outdir=outdir,
        checkpoint_freq=args.checkpoint_freq,
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        max_episode_len=TIME_LIMIT,
        evaluation_hooks=[WandBLoggerBatch()]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('policy', choices=['ppo', 'sac'], help='what policy to use')
    parser.add_argument(
        "outdir",
        type=str,
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument('--checkpoint-freq', type=int, default=50_000)
    parser.add_argument(
        "--env-name",
        type=str,
        default="dmc:Cheetah-run-v1",
        choices=[f'dmc:{env}-v1' for env in ['Cheetah-run', 'Walker-walk', 'Pendulum-swingup', 'Cartpole-swingup']],
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    # parser.add_argument(
    #     "--replay-start-size",
    #     type=int,
    #     default=10000,
    #     help="Minimum replay buffer size before " + "performing gradient updates.",
    # )
    # parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    # parser.add_argument(
    #     "--policy-output-scale",
    #     type=float,
    #     default=1.0,
    #     help="Weight initialization scale of policy output.",
    # )
    args = parser.parse_args()

    env_id = args.env_name.split(":")[-1].lower()

    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project='maps-pfrl',
        group=env_id,
        config=vars(args),
    )

    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     avail_gpus = [0, 1, 2, 3]
    #     # gpu_id = 0 if args.line_number is None else args.line_number % len(avail_gpus)
    #     gpu_id = 1
    #     cvd = avail_gpus[gpu_id]
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    outdir = Path(args.outdir) / env_id / args.policy / wandb.run.project / wandb.run.id
    outdir.mkdir(mode=0o775, parents=True, exist_ok=True)

    main(args, outdir)
