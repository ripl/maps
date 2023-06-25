#!/usr/bin/env python3
import sys
from pathlib import Path

# HACK to work with python import
proj_rootdir = Path(__file__).resolve().parents[3]
sys.path.append(str(proj_rootdir))

import json
from maps.scripts.pretraining.experts import (
    cheetah_ppo,
    cheetah_sac,
    walker_ppo,
    walker_sac,
    pendulum_ppo,
    pendulum_sac,
    cartpole_ppo,
    cartpole_sac,
)


def convert(model_infos):
    return [(minfo['policy'], minfo['path']) for minfo in model_infos]


cheetah_ppo = convert(cheetah_ppo)
cheetah_sac = convert(cheetah_sac)
walker_ppo = convert(walker_ppo)
walker_sac = convert(walker_sac)
pendulum_ppo = convert(pendulum_ppo)
pendulum_sac = convert(pendulum_sac)
cartpole_ppo = convert(cartpole_ppo)
cartpole_sac = convert(cartpole_sac)


defaults = {
    # "std_from_means": True,
    "ase_sigma_ratio": 0.5,
}


def create_sweep(fname, envs, env2experts_list, env2ase_sigma, algorithms, learner_pis, seeds, pggae=False):
    # mamba and lops-aps
    lines = []
    for seed in seeds:
        for env in envs:
            env_name = f'dmc:{env}-v1'
            print('env', env, 'env_name', env_name)
            for experts in env2experts_list[env]:
                for learner_pi, algorithm in zip(learner_pis, algorithms):
                    ase_sigma = env2ase_sigma[env] if algorithm == "maps-se" else 0.
                    lines.append(
                        {
                            "env_name": env_name,
                            "experts_info": experts,
                            "algorithm": algorithm,
                            "use_riro_for_learner_pi": learner_pi,
                            "ase_sigma": ase_sigma,
                            "seed": seed,
                            **defaults
                        }
                    )
            if pggae:
                # For pg-gae, the experts don't matter
                expert_paths = {
                    'cheetah-run': cheetah_ppo[:1],
                    'cartpole-swingup': cartpole_ppo[:1],
                    'walker-walk': walker_ppo[:1],
                    'pendulum-swingup': pendulum_ppo[:1]
                }[env.lower()]

                lines.append(
                    {
                        "env_name": env_name,
                        "experts_info": expert_paths,
                        "algorithm": "pg-gae",
                        "use_riro_for_learner_pi": "none",
                        "ase_sigma": 0,
                        "seed": seed,
                        **defaults
                    }
                )

    json_text = [json.dumps(line, sort_keys=True) for line in lines]
    print(f'{len(json_text)} lines to {fname}')
    with open(fname, 'w') as f:
        f.write('\n'.join(json_text))


if __name__ == '__main__':
    import sys
    this_file_name = sys.argv[0]

    # Variables to sweep over
    envs = ['Cheetah-run', 'Walker-walk', 'Pendulum-swingup', 'Cartpole-swingup']

    # Set of experts to work with
    env2experts_list = {
        'Cheetah-run': [cheetah_ppo[:3], cheetah_ppo[::4][:3], cheetah_sac[:3], cheetah_sac[::4][:3], cheetah_sac[-3:]],
        'Walker-walk': [walker_ppo[:3], walker_ppo[::4][:3], walker_sac[:3], walker_sac[::4][:3], walker_sac[-3:]],
        'Pendulum-swingup': [pendulum_ppo[:3], pendulum_ppo[::4][:3], pendulum_sac[:3], pendulum_sac[-3:]],
        'Cartpole-swingup': [cartpole_ppo[:3], cartpole_ppo[::4][:3], cartpole_sac[:3], cartpole_sac[::4][:3], cartpole_sac[-3:]]}

    # Sigma for active state exploration
    env2ase_sigma = {
        'Cheetah-run': 2.5,
        'Walker-walk': 10,
        'Pendulum-swingup': 0.25,
        'Cartpole-swingup': 0.25,  # <-- We should run a sweep to find out a good value for this
    }

    seeds = [i for i in range(5)]
    learner_pis = ['rollin', 'all', 'all']  # Whether the learner uses expert rollout for training or not
    algorithms = ['mamba', 'maps', 'maps-se']

    fname = Path(this_file_name).stem + '.jsonl'
    create_sweep(fname, envs, env2experts_list, env2ase_sigma, algorithms, learner_pis, seeds, pggae=True)
