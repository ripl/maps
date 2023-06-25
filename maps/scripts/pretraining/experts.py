#!/usr/bin/env python3


# NOTE: This script assumes that pretrained experts are stored under `/maps/experts`
def get_entry(env, policy, wandb_id, step):
    ckpt = '/maps/experts/{env}/{policy}/alops-pfrl/{wandb_id}/{step}_checkpoint'
    path = ckpt.format(env=env, policy=policy, wandb_id=wandb_id, step=step)
    return {'path': path, 'policy': policy, 'step': step}


# Every 100k steps up to 1.5M step (https://wandb.ai/takuma-yoneda/alops-pfrl-cheetah-run-v1/runs/e1snmaby?workspace=user-takuma-yoneda)
steps = [int(100 * 1e3 * (i+1)) for i in range(15)]
small_steps = [int(100 * 1e3 * (i+1)) for i in range(8)]
cheetah_ppo = [get_entry(env='cheetah-run-v1', policy='ppo', wandb_id='5u59suv0', step=step) for step in steps]
walker_ppo = [get_entry(env='walker-walk-v1', policy='ppo', wandb_id="ch974g0h", step=step) for step in steps]
# pendulum_ppo = [get_entry(env='pendulum-swingup-v1', policy='ppo', wandb_id="4qylv6es", step=step) for step in steps]
pendulum_ppo = [get_entry(env='pendulum-swingup-v1-xuefeng', policy='ppo', wandb_id="76mnfikb", step=step) for step in steps]
cartpole_ppo = [get_entry(env='cartpole-swingup-v1', policy='ppo', wandb_id="2o5flp32", step=step) for step in steps]

cheetah_sac = [get_entry(env='cheetah-run-v1', policy='sac', wandb_id='dn10zjdw', step=step) for step in steps]
walker_sac = [get_entry(env='walker-walk-v1', policy='sac', wandb_id='6azmiwpz', step=step) for step in steps]
# pendulum_sac = [get_entry(env='pendulum-swingup-v1', policy='sac', wandb_id='ludvoyav', step=step) for step in steps]
pendulum_sac = [get_entry(env='pendulum-swingup-v1-xuefeng', policy='sac', wandb_id='en1olu45', step=step) for step in small_steps]
cartpole_sac = [get_entry(env='cartpole-swingup-v1', policy='sac', wandb_id='1bjvz6gl', step=step) for step in steps]
