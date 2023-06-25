#!/usr/bin/env python3
"""This script evaluates each expert and report their performance in a table.
"""

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
import os
from typing import Tuple
import functools
from alops import logger
from alops.helpers.env import rollout_single_ep, make_env
from alops.agents.expert import ExpertAgent
from alops.scripts.train import Evaluator
import pandas as pd
import gym
import math
from .experts import (
    cheetah_ppo,
    cheetah_sac,
    walker_ppo,
    walker_sac,
    pendulum_ppo,
    pendulum_sac,
    cartpole_ppo,
    cartpole_sac,
)
max_episode_len = 1000


env2modelpaths = {
    'dmc:Cheetah-run-v1': cheetah_ppo + cheetah_sac,
    'dmc:Walker-walk-v1': walker_ppo + walker_sac,  # Performance collapses after 1.5M steps.
    'dmc:Pendulum-swingup-v1': pendulum_ppo + pendulum_sac,
    'dmc:Cartpole-swingup-v1': cartpole_ppo + cartpole_sac,
}


def save_video(frame_stack, path, fps=20, **imageio_kwargs):
    """Save a video from a list of frames.

    Correspondence: https://github.com/geyang/ml_logger
    """
    import os
    import tempfile, imageio  # , logging as py_logging
    import shutil
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    format = 'mp4'
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        from skimage import img_as_ubyte
        try:
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        ntp.seek(0)
        shutil.copy(ntp.name, path)


def run_eval(conn: Connection, clean_envname: str, policy_type: str, step: int, model_path: str, sample_env: gym.Env, evaluator: Evaluator, num_eval_episodes: int, outdir: Path):
    """Run evaluation and send the result back to a caller process
    """

    # NOTE: stdout/stderr in a subprocess often clogs the Pipe.
    # Redirect stdout/stderr to files
    import sys
    # sys.stdout = open(os.devnull, 'w')
    # sys.stderr = open(os.devnull, 'w')

    expert = ExpertAgent(sample_env, model_path, policy=policy_type)
    logger.info(f'Evaluating {policy_type} at step {step} (path: {model_path})')
    logs = evaluator.evaluate(expert, num_eval_episodes=num_eval_episodes, save_video_num_ep=5)
    logger.info(f'Evaluation is done ({policy_type}, {step})')
    ret_mean, ret_std = logs['eval/returns_mean'], logs['eval/returns_std']

    row = {'env': clean_envname,
           'policy': policy_type,
           'step': step,
           'score': f'{ret_mean:.1f} &plusmn; {ret_std:.1f}',
           # 'video': '<video src="....mp4" width=180 />',  # TODO: fill this in
           'path': model_path,
           # '_ret_mean': float(ret_mean),
           # '_ret_std': float(ret_std)
           }

    # Save video (logs['frames'])
    video_path = outdir / 'videos' / clean_envname / f'{policy_type}-{step:07d}.mp4'
    video_path.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
    logger.info('saving video...')
    save_video(logs['eval/frames'], video_path)
    logger.info('saving video...done')

    logger.info('Sending message!')
    conn.send(row)
    conn.close()


def main(env2modelpaths, outdir):
    num_eval_episodes = 30
    outdir = Path(outdir)
    outdir.mkdir(mode=0o755, parents=True, exist_ok=True)

    df = pd.DataFrame(columns=['env', 'policy', 'step', 'score', 'path'])

    # Verify if model paths are valid
    for _, models in env2modelpaths.items():
        for model in models:
            if not Path(model['path']).is_dir():
                raise ValueError(f'directory not found: {model["path"]}')

    for env_name, models in env2modelpaths.items():
        clean_envname = env_name.split(':', maxsplit=1)[-1].lower()
        mkenv = functools.partial(make_env, env_name, seed=0, test=True)
        sample_env = mkenv()
        evaluator = Evaluator(mkenv, max_episode_len)

        # TODO: When the number of models is too large, they don't fit in a single GPU...
        # Let's create batches
        minibatch_size = 10
        num_minibatches = math.ceil(len(models) / minibatch_size)
        minibatch_models = [models[i * minibatch_size: (i + 1) * minibatch_size] for i in range(num_minibatches)]

        for models in minibatch_models:
            logger.info(f'Running {len(models)} models in parallel...')
            # Run evaluation for models in parallel.
            processes = []
            for model in models:
                parent_conn, child_conn = Pipe()
                process = Process(
                    target=run_eval,
                    args=(child_conn, clean_envname, model['policy'], model['step'], model['path'], sample_env, evaluator, num_eval_episodes, outdir)
                )
                processes.append((process, parent_conn, child_conn))

            for p, pconn, _ in processes:
                p.start()

            for p, pconn, child_conn in processes:
                print('Trying to receive the message...')
                row = pconn.recv()
                print(f'received: {row}')
                df = pd.concat((df, pd.DataFrame([row])))

            for p, pconn, cconn in processes:
                p.join()

    markdown_path = outdir / 'experts-summary.md'
    logger.info(f'Saving a markdown file to {markdown_path}...')

    # Sort the table based on policy, env, and step
    df = df.sort_values(by=['env', 'policy', 'step'])
    mdtxt = f'# Expert performances on each environment (`num-eval-steps = {num_eval_episodes}`)\n\n'
    mdtxt += df.to_markdown()
    with open(markdown_path, 'w') as f:
        f.write(mdtxt)

    df.to_csv(outdir / 'experts-summary.csv')


if __name__ == '__main__':
    from pfrl.utils import set_random_seed
    seed = 42

    # Set a random seed used in PFRL
    set_random_seed(seed)

    # outdir = os.environ['RMX_OUTPUT_DIR']
    outdir = '/lops/experts/summary'
    # NOTE: Checkpoints are saved every 50k steps





    # env2modelpaths = {'dmc:Cheetah-run-v1': cheetah_ppo}
    main(env2modelpaths, outdir)
