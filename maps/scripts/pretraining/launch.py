#!/usr/bin/env python3

steps = 500_000
num_envs = 32
outdir = '$RMX_OUTPUT_DIR/short'
checkpoint_freq = 10_000
cmds = [
    f"CUDA_VISIBLE_DEVICES=1 python -m alops.scripts.pretraining.train_expert ppo --env-name dmc:Cheetah-run-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=2 python -m alops.scripts.pretraining.train_expert ppo --env-name dmc:Walker-walk-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=3 python -m alops.scripts.pretraining.train_expert ppo --env-name dmc:Pendulum-swingup-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=1 python -m alops.scripts.pretraining.train_expert ppo --env-name dmc:Cartpole-swingup-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=1 python -m alops.scripts.pretraining.train_expert sac --env-name dmc:Cheetah-run-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=2 python -m alops.scripts.pretraining.train_expert sac --env-name dmc:Walker-walk-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=3 python -m alops.scripts.pretraining.train_expert sac --env-name dmc:Pendulum-swingup-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
    f"CUDA_VISIBLE_DEVICES=1 python -m alops.scripts.pretraining.train_expert sac --env-name dmc:Cartpole-swingup-v1 --steps {steps} --num-envs {num_envs} --outdir {outdir} --checkpoint-freq {checkpoint_freq}",
]

if __name__ == "__main__":
    import argparse
    import subprocess
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("line_number", type=int, help="line number")
    args = parser.parse_args()

    avail_gpus = [1, 2, 3]
    envvars = os.environ.copy()
    envvars['CUDA_VISIBLE_DEVICES'] = str(avail_gpus[args.line_number % len(avail_gpus)])

    print(f'running {cmds[args.line_number]}...')
    subprocess.call(cmds[args.line_number], shell=True, env=envvars)
    print(f'running {cmds[args.line_number]}...done')
