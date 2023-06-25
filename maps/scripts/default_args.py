#!/usr/bin/env python3
from params_proto import ParamsProto
from params_proto import Proto


class Args(ParamsProto):
    algorithm = 'maps-se'  # [lops-aps-ase, lops-aps, mamba, pg-gae, aggrevated]
    num_train_steps = 100
    gamma = .995
    lmd = 0.9

    experts_info = []  # [('sac', path-to-model), ('ppo', path-to-model), blah]

    max_grad_norm = 1.0

    num_rollouts = 8
    num_eval_episodes = 32
    num_epochs = 2
    seed = 0
    use_ase_sigma_coef = False
    ase_sigma = 10.
    use_ase_sigma_ratio = True
    ase_sigma_ratio = 0.5
    ase_sigma_coef = 1.
    batch_size = 128

    expert_vfn_gain = 1.0
    num_expert_vfns = 5

    pret_num_rollouts = 8
    pret_num_epochs = 32
    pret_num_val_iterations = 32

    learner_buffer_size = 2048
    expert_buffer_size = 19200

    use_expert_obsnormalizer = True

    # Compute stddev and variance from predicted means.
    # Prediction of stddev is ignored if set to True
    std_from_means = True

    deterministic_experts = False
    experts_dir = Proto(env='EXPERTS_DIR')

    env_name = 'dmc:Cheetah-run-v1'
    max_episode_len = 1000
    use_riro_for_learner_pi = 'none'  # 'none', 'rollin', 'all'
