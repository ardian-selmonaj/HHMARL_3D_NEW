import os
import glob
import time
import torch
import tyro
import numpy as np
from args import Args
from algorithm.networks import build_actor_critic, act_with_policies
from env.env import AircombatMARL
from rendering.flightsim import FlightSimClosed
from train import process_args

RENDER = True
RENDERER = "flightsim"  # "pygame": simple rendering of this repository, "flightsim": FlightSim (see README)
RESTORE_PATH = r"runs\cannon_L4_f16"
TASK = "cannon"
LEVEL = 4

# Initial states of the scenarios, per aircraft relative to the battle field center: "north"/"east" in m (or "lat"/"lon"
# in degrees), "alt" in m above sea level, "speed" in m/s, "heading" in degrees (0 north, 90 east), and optionally
# "roll"/"pitch" in degrees. B1 (the RL agent, followed by the camera) starts at the center heading north; R2 (the opponent)
# is placed for the WEZ signals of FlightSim in the cockpit and follow views (F1, F2). The weapon engagement zone (WEZ) of
# the cannon is within 10 deg of the flight direction and 3500 m.
SCENARIOS = {
    # B1 2 km behind R2, both heading north: R2 is in the WEZ of B1 -> green circle
    "agent_faces_opponent": {"B1": {"north": 0, "east": 0, "alt": 6000, "heading": 0, "speed": 250},       # m, m, m, deg, m/s
                             "R2": {"north": 2000, "east": 0, "alt": 6000, "heading": 0, "speed": 230}},
    # Front vs. front 2.5 km apart, 150 m sideways (3.4 deg off both noses): each in the WEZ of the other -> green circle and red beam
    "head_on": {"B1": {"north": 0, "east": 0, "alt": 6000, "heading": 0, "speed": 250},
                "R2": {"north": 2500, "east": 150, "alt": 6000, "heading": 180, "speed": 250}},
    # R2 2 km behind B1, 30 deg left of its tail, heading at B1, which flies away from it: B1 is in the WEZ of R2 -> red beam
    # from the left edge of the view (closer to the tail it turns into a red border around the view, e.g. straight behind with
    # "east": 0 and "heading": 0)
    "opponent_behind": {"B1": {"north": 0, "east": 0, "alt": 6000, "heading": 0, "speed": 250},
                        "R2": {"north": -1732, "east": -1000, "alt": 6000, "heading": 30, "speed": 250}},
}
SCENARIO = "head_on"           # one of SCENARIOS for every episode, None: all of them one after another
# FlightSim shows the initial state of each episode this long (s) before the policies fly, so that the signals can be seen
# (meanwhile F1/F2 switch the view and Z the signals): once an aircraft fires in a WEZ, it usually hits within a second.
HOLD_SECONDS = 0.0

# FlightSim settings (RENDER = True and RENDERER = "flightsim")
JSBSIM_STATES = "python"  # "python": the JSBSim of this env simulates the aircraft, "c++": the JSBSim of FlightSim
MANUAL = False            # True: a human flies MANUAL_AGENT with the keyboard, as teammate of the blue RL agents
MANUAL_AGENT = "B2"       # added as copy of B1 if the config has no such aircraft (track/cannon/escape), replaces it otherwise (hier: B3)
LOCATION = (8.965318, 45.975305)  # (longitude, latitude) of the battle field center in degrees, here Lugano
MINIMAP = True            # map of the aircraft around the camera target in the bottom right corner (key M toggles it, N north-up/heading-up)
MINIMAP_RANGE = 20000     # m, radius of the minimap; aircraft farther away are shown at its edge with their distance
WEZ = True                # cockpit/follow view (F1/F2): green circle if the camera target can hit an opponent, red beam if an opponent can hit it (key Z)
CAMERA_VIEW = "Follow"    # view at the start of each episode: "Cockpit" (F1) or "Follow" (F2) show the WEZ signals, "Overview" (F3), "Free" (F4)

def hold_initial_state(env, seconds):
    """Shows the initial state in FlightSim in CAMERA_VIEW for the given time, without advancing the simulation."""
    app = env.env.flightsim.app
    # Selecting the view again (via another one) turns the follow camera at once to the new heading of the aircraft,
    # otherwise it turns there only while the simulation runs
    app.select_camera_view("Follow" if CAMERA_VIEW == "Cockpit" else "Cockpit")
    app.select_camera_view(CAMERA_VIEW)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.step(0)  # updates the camera and the signals without advancing the simulation
        if not app.render():
            raise FlightSimClosed("FlightSim window closed")

def load_checkpoint(restore_path, device):
    path = restore_path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "checkpoint*.pt")))
        assert candidates, f"No checkpoints found in {path}"
        path = candidates[-1]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {path}  (trained to step {ckpt['step']})")
    return ckpt

def run_inference(num_episodes=500):
    args, _, device = process_args(tyro.cli(Args))
    args.restore_path = RESTORE_PATH
    args.task = TASK
    args.level = LEVEL
    
    if args.hierarchy_level == 2:
        args.level = 1
        args.task = "hier"
    else:
        args.hierarchy_level = 1
        args.task = f"{args.task}/L{args.level}"

    kwargs = {"hierarchy_level":args.hierarchy_level, "training_level": args.level+int(args.task == "track" or args.task == "escape" or args.hierarchy_level == 2), 
              "rendering": RENDER, "render_view": 0, "render_samples": args.render_samples}

    assert RENDERER in ("pygame", "flightsim"), f"Unknown RENDERER {RENDERER}"
    use_flightsim = RENDER and RENDERER == "flightsim"
    assert use_flightsim or (not MANUAL and JSBSIM_STATES == "python"), \
        "MANUAL and JSBSIM_STATES = 'c++' need RENDER = True and RENDERER = 'flightsim'"
    if use_flightsim:
        kwargs["flightsim"] = {"jsbsim_states": JSBSIM_STATES, "manual_agent": MANUAL_AGENT if MANUAL else None,
                               "minimap": MINIMAP, "minimap_range": MINIMAP_RANGE, "wez": WEZ, "camera_view": CAMERA_VIEW}
        kwargs["config_overrides"] = {"battle_field_center": [LOCATION[0], LOCATION[1], 0.0]}

    assert args.restore_path is not None, "Set --restore_path to a checkpoint file or directory."

    env = AircombatMARL(task=args.task, **kwargs)
    actors, _ = build_actor_critic(
        args, env.get_obs_size(), env.get_action_space(),
        env.get_state_size(), env.n_agents, device,
    )
    actors.load_state_dict(load_checkpoint(args.restore_path, device)["actors_state_dict"])
    actors.eval()

    agent_keys = env.agent_keys
    A = env.n_agents
    L, H = args.rnn_num_layers, args.actor_hidden_dim
    assert SCENARIO is None or SCENARIO in SCENARIOS, f"Unknown SCENARIO {SCENARIO}, available: {list(SCENARIOS)}"
    scenarios = [SCENARIO] if SCENARIO else list(SCENARIOS)

    ep_rewards, ep_lengths = [], []
    wins = losses = 0
    try:
        for ep in range(num_episodes):
            scenario = scenarios[ep % len(scenarios)]
            obs_dict, _ = env.reset(seed=args.seed + ep, options={"init_states": SCENARIOS[scenario]})
            obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)
            print(f"Episode {ep + 1:3d} | scenario: {scenario}")
            if use_flightsim and HOLD_SECONDS > 0:
                hold_initial_state(env, HOLD_SECONDS)

            rnn_hxs = torch.zeros(1, A, L, H, device=device) if args.rnn else None
            masks = torch.ones(1, device=device) if args.rnn else None

            ep_reward, ep_len, done = 0.0, 0, False
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

                if RENDER and not args.hierarchy_level == 2:
                    env.env.render()

            agents = env.env.agents
            n_killed_opp = sum(not agents[aid].alive for aid in env.env.opp_agent_ids)
            n_killed_agent = sum(not agents[aid].alive for aid in env.env.rl_agent_ids)

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            n_opp = len(env.env.opp_agent_ids)
            n_rl = len(env.env.rl_agent_ids)
            print(f"Episode {ep + 1:3d} | reward: {ep_reward:8.3f} | length: {ep_len:4d} | killed opp: {n_killed_opp} | killed agent: {n_killed_agent}")
            if n_killed_opp == n_opp:
                wins += 1
            if n_killed_agent == n_rl:
                losses += 1
    except FlightSimClosed:
        print("FlightSim window closed.")

    if ep_rewards:
        n = len(ep_rewards)
        print(f"\nMean reward: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}  "
              f"| Mean length: {np.mean(ep_lengths):.1f}")
        print(f"Mean Wins: {wins/n:.3f} | Mean Losses: {losses/n:.3f}")
    env.close()


if __name__ == "__main__":
    run_inference()
