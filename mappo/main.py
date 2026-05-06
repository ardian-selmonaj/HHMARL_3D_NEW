import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import datetime
import torch
import tyro
import random
import numpy as np
from tqdm import tqdm
from args import Args
from learner import Learner
from utils import validate_decay_schedule, get_decayed_value, Logger, Workers

"""
The main file for running experiments on AircraftEnv.
"""

RUN_NAME = "rew2"

def process_args(args):
    kwargs = {"hierarchy_level":args.hierarchy_level, "training_level": 2}

    args.env_name = "track"
    args.level = 1

    #args.restore_path = 
    args.run_name = f"{RUN_NAME}"
    args.exp_name = f"{args.env_name}_L{args.level}"

    args.env_name = f"{args.env_name}/L{args.level}_{args.env_name}"
    args.num_envs = 20
    args.batch_size = 10000
    args.train_epochs = 10
    args.ppo_clip = 0.25
    args.minibatch_size = 5000
    args.learning_rate_actor = 3e-4
    args.learning_rate_critic = 3e-4
    args.actor_hidden_dim = 128
    args.critic_hidden_dim = 128
    args.rnn = False

    args.spo = False
    args.use_ReLU = True
    args.td_lambda = 0.92
    args.entropy_decay = [[0, 0.1], [3e6, 0]]
    args.total_timesteps = 20000000

    # cannon: L1: -20M, L1.5: -40M, L2: -80M, L3: 120M || track: L1: -20M, L2: -60M
    # L1 entropy decay [[0, 0.1], [3e6, 0]]
    # L1.5 restore: [[20e6, 0.1], [20.1e6, 0]]
    # L2 restore: [[40e6, 0.1], [40.1e6, 0]]
    # L3 restore: [[80e6, 0.1], [80.1e6, 0]]

    args.avail_actions = False
    args.torch_compile = False
    args.cuda_id = 0

    batch_alignment = math.lcm(args.num_envs, args.minibatch_size)
    args.batch_size = int(math.ceil(args.batch_size / batch_alignment) * batch_alignment)
    args.lr_decay_schedule = validate_decay_schedule(args.lr_decay, "lr_decay")
    args.entropy_decay_schedule = validate_decay_schedule(args.entropy_decay, "entropy_decay")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(args.cuda_id)
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    torch.backends.cudnn.deterministic = args.torch_deterministic

    return args, kwargs, device


def restore_checkpoint(args, device, learner):
    """Load a checkpoint to resume training. Returns the step to resume from."""
    restore = args.restore_path
    if os.path.isdir(restore):
        restore = os.path.join(restore, "checkpoint.pt")
    assert os.path.isfile(restore), f"No checkpoint found at {restore}"
    ckpt = torch.load(restore, map_location=device)
    learner.actors.load_state_dict(ckpt["actors_state_dict"])
    learner.critic.load_state_dict(ckpt["critic_state_dict"])
    learner.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
    learner.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
    if ckpt["value_normalizer_state_dict"] is not None and learner.value_normalizer is not None:
        learner.value_normalizer.load_state_dict(ckpt["value_normalizer_state_dict"])
    print(f"Restored checkpoint from {restore} at step {ckpt['step']}")
    return ckpt["step"]


if __name__ == "__main__":
    args, kwargs, device = process_args(tyro.cli(Args))

    workers = Workers(args, kwargs)
    learner = Learner(args, device, workers)
    logger = Logger(log_dir=f"runs/{args.exp_name}_{args.run_name}", args=args)

    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_episodes = 0
    update_step = 0
    step = restore_checkpoint(args, device, learner) if args.restore_path is not None else 0
    iteration_times = []

    progress_bar = tqdm(total=args.total_timesteps, initial=step, desc=f"train {args.exp_name} {args.env_name}", dynamic_ncols=True)

    while step < args.total_timesteps:
        iteration_start = time.time()

        # 1. Update scheduled hyperparameters
        current_lr = get_decayed_value(step, args.learning_rate_actor, args.lr_decay_schedule)
        current_critic_lr = get_decayed_value(step, args.learning_rate_critic, args.lr_decay_schedule)
        current_entropy_coef = get_decayed_value(step, args.entropy_coef, args.entropy_decay_schedule)
        learner.set_learning_rates(current_lr, current_critic_lr)

        # 2. Collect rollout
        rollout = learner.collect_rollout(step)
        step += rollout.steps_collected
        progress_bar.update(rollout.steps_collected)

        ep_rewards.extend(rollout.ep_rewards)
        ep_lengths.extend(rollout.ep_lengths)
        ep_stats.extend(rollout.ep_stats)
        num_episodes += rollout.num_new_episodes

        if logger.log_episode_stats(ep_rewards, ep_lengths, ep_stats, num_episodes, step):
            ep_rewards, ep_lengths, ep_stats = [], [], []
            logger.save_checkpoint(learner, step)

        # 3. Compute GAE advantages and flatten batch
        flat_data = learner.compute_advantages()

        # 4. Train actor and critic on the collected batch
        metrics = learner.train(flat_data, current_entropy_coef)
        update_step += 1

        logger.log_train_stats(
            metrics, learner.training_step, update_step,
            current_lr, current_critic_lr, current_entropy_coef, step
        )

        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        mean_reward = (
            float(np.mean(rollout.iteration_rewards))
            if rollout.iteration_rewards
            else float(np.mean(learner._current_ep_rewards))
        )
        progress_bar.set_postfix(
            training_step=learner.training_step,
            mean_reward=f"{mean_reward:.3f}",
            mean_iter_s=f"{float(np.mean(iteration_times)):.3f}",
        )

    progress_bar.close()
    workers.close()
    logger.close()
