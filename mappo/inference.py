import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tyro
import numpy as np

from args import Args
from networks import build_actor_critic, act_with_policies
from envs import get_environment
from main import process_args

RENDER = True

def load_checkpoint(restore_path, device):
    path = restore_path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "checkpoint*.pt")))
        assert candidates, f"No checkpoints found in {path}"
        path = candidates[-1]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {path}  (trained to step {ckpt['step']})")
    return ckpt

TASK = "escape"
LEVEL = 1
PATH = "/home/ardian/projects/HHMARL_3D_NEW/runs_verify/escape/escape_L1_mars_dscale2_hscale005_RNN"

def run_inference(num_episodes=10):
    # Reuse the exact training config so the network architecture matches the checkpoint.
    args, _, device = process_args(tyro.cli(Args))

    args.restore_path = PATH
    args.level = LEVEL
    args.env_name = f"{TASK}/L{LEVEL}_{TASK}"
    args.env_type = "aircombat"
    kwargs = {"hierarchy_level":args.hierarchy_level, "training_level": args.level+int(TASK == "track" or TASK == "escape")}

    args.actor_hidden_dim = 128
    args.critic_hidden_dim = 128
    args.rnn = False
    args.use_ReLU = True
    args.avail_actions = False
    args.torch_compile = False

    assert args.restore_path is not None, "Set --restore_path to a checkpoint file or directory."

    env = get_environment(args.env_type, args.env_name, args.agent_ids, kwargs)
    actors, _ = build_actor_critic(
        args, env.get_obs_size(), env.get_action_space(),
        env.get_state_size(), env.n_agents, device,
    )
    actors.load_state_dict(load_checkpoint(args.restore_path, device)["actors_state_dict"])
    actors.eval()

    agent_keys = env.agent_keys
    A = env.n_agents
    L, H = args.rnn_num_layers, args.actor_hidden_dim

    ep_rewards, ep_lengths = [], []
    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=args.seed + ep)
        obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)

        rnn_hxs = torch.zeros(1, A, L, H, device=device) if args.rnn else None
        masks = torch.ones(1, device=device) if args.rnn else None

        ep_reward, ep_len, done = 0.0, 0, False
        killed_agent = "--"
        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)  # (1, A, D)
            with torch.no_grad():
                actions, _, _, rnn_hxs = act_with_policies(
                    args, actors, obs_t, rnn_hxs=rnn_hxs, masks=masks, deterministic=True,
                )
            actions_np = actions[0].cpu().numpy()  # (A, K)
            actions_dict = {k: actions_np[i] for i, k in enumerate(agent_keys)}

            obs_dict, reward_dict, done_dict, trunc, info = env.step(actions_dict)
            obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)
            ep_reward += float(np.mean([reward_dict[k] for k in agent_keys]))
            ep_len += 1
            done = done_dict["all"] or trunc["all"]

            if RENDER:
                env.env.render()

            if info["opp_killed"]:
                killed_agent = "opp"
            if info["agent_killed"]:
                killed_agent = "agent"

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        print(f"Episode {ep + 1:3d} | reward: {ep_reward:8.3f} | length: {ep_len:4d} | killed: {killed_agent}")

    print(f"\nMean reward: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}  "
          f"| Mean length: {np.mean(ep_lengths):.1f}")
    env.close()


if __name__ == "__main__":
    run_inference()
