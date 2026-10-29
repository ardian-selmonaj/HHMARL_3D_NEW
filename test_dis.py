"""Deployment of trained policies with and against the aircraft of VR-Forces over DIS (IEEE 1278.1), see the README.

The RL agents (and optional simulated opponents) are simulated by the JSBSim of the env and sent to VR-F, the VR-F aircraft are
received over DIS and observed by the policies like the aircraft in training. The env runs in real time.
Needs opendis: pip install --no-deps opendis==1.0
"""
import time
import warnings
import torch
import tyro
import numpy as np
from args import Args
from algorithm.networks import build_actor_critic, act_with_policies
from env.env import AircombatMARL
from env.utils import parse_config
from test import load_checkpoint
from train import process_args

VRF_SIMULATION = False  # True: all aircraft, also the RL agents, are simulated by VR-F and only the actions are computed here
                        # (not implemented yet). False: the RL agents are simulated by the JSBSim of this env, the VR-F aircraft by VR-F.

# DIS network, as set in VR-Forces
DIS_ADDRESS = "10.11.255.255"  # broadcast address of the network of VR-F (or the IP address of its computer, or a multicast group)
DIS_PORT = 3000                # UDP port
EXERCISE_ID = 1                # DIS exercise ID of the VR-F scenario
SITE_ID = 1                    # DIS site and application ID of this program, the application ID must differ from the one of VR-F
APPLICATION_ID = 8492

# Policy, as in test.py (the commander of a hierarchical run with: python test_dis.py --hierarchy_level 2)
RESTORE_PATH = r"runs\cannon_L4_f16"
TASK = "cannon"
LEVEL = 4

# Aircraft. Team B: the RL agents and VR-F teammates. Team R: the simulated and VR-F opponents.
NUM_RL_AGENTS = 1            # simulated by the JSBSim of this env and flown by the policy
VRF_TEAMMATES = []           # markings of the VR-F aircraft flying with the RL agents, e.g. ["F-16C 1"]; an entry None takes any VR-F aircraft of the friendly force
VRF_OPPONENTS = ["F-35A 1"]  # markings of the VR-F aircraft fought by the RL agents, e.g. ["F-35A 1", "F-22 1"]; an entry None takes any VR-F aircraft of the opposing force
NUM_SIM_OPPONENTS = 0        # opponents simulated by the JSBSim of this env, with the opponent mission of the task (e.g. model_strategy for cannon L4)
LOCATION = (7.6280, 46.7580)  # (longitude, latitude) of the battle field center in degrees, in the area of the VR-F scenario. The observations
                              # and the map boundary (map_size) are relative to it, as in training.

# Initial states of the aircraft of this env, as in test.py: None for the states of the config (random if rnd_init) around LOCATION,
# or per aircraft relative to LOCATION: "north"/"east" in m (or "lat"/"lon" in degrees), "alt" in m, "speed" in m/s, "heading" in degrees.
# The VR-F aircraft are where VR-F flies them.
INIT_STATES = None
# INIT_STATES = {"B1": {"north": -6000, "east": 0, "alt": 6000, "heading": 0, "speed": 250}}  # m, m, m, deg, m/s
HORIZON = None               # steps per episode, None for the horizon of the config (low-level: 600 steps of 0.13 s)
NUM_EPISODES = 10

# Hits. A VR-F detonation within KILL_RADIUS of an aircraft of this env hits it with the hit probability of the cannon of the env
# (80 %). A cannon hit of this env on a VR-F aircraft is sent to VR-F as detonation of MUNITION_TYPE, so that VR-F destroys it.
KILL_RADIUS = 50.0                       # m
MUNITION_TYPE = [2, 1, 225, 1, 2, 3, 0]  # DIS entity type: AIM-120C (munition, anti-air, USA, guided, AMRAAM, C)
MUNITION_WARHEAD = 1620                  # DIS enumeration: high explosive, blast fragmentation
MUNITION_FUSE = 3000                     # DIS enumeration: proximity
ENTITY_TYPES = {"f16": [1, 2, 225, 1, 2, 1, 0], "f15": [1, 2, 225, 1, 5, 1, 0]}  # DIS entity types of the aircraft sent to VR-F per JSBSim model (others: the first)

# DIS details
SEND_RATE = 25          # Hz, entity state PDUs of each aircraft of this env
ENTITY_TIMEOUT = 12.0   # s without entity state PDU after which a VR-F aircraft counts as removed
WAIT_TIMEOUT = None     # s to wait for the VR-F aircraft before each episode, None: as long as needed
LIST_ENTITIES = False   # only list the entities received over DIS (IDs, markings, forces, types, positions), e.g. to find the markings
SHOW_STATES = True      # print the states of the RL agents and the DIS traffic every STATUS_INTERVAL seconds
STATUS_INTERVAL = 2.0


def dis_options(vrf_entities):
    """Arguments of comm.dis_bridge.DISBridge."""
    return {"address": DIS_ADDRESS, "port": DIS_PORT, "exercise_id": EXERCISE_ID, "site_id": SITE_ID,
            "application_id": APPLICATION_ID, "vrf_entities": vrf_entities, "entity_types": ENTITY_TYPES,
            "munition": (MUNITION_TYPE, MUNITION_WARHEAD, MUNITION_FUSE), "send_rate": SEND_RATE,
            "entity_timeout": ENTITY_TIMEOUT, "kill_radius": KILL_RADIUS, "wait_timeout": WAIT_TIMEOUT}


def roster(task):
    """The aircraft of the env: the RL agents and VR-F teammates (team B), then the simulated and VR-F opponents (team R),
    numbered across the teams as in the hier config. They copy the RL agent and the opponent of the config of the task, whose
    other aircraft are removed. @returns the aircraft for the config overrides and the markings of the VR-F aircraft."""
    config_aircraft = parse_config(task).aircraft
    rl_agent = next(config for config in config_aircraft.values() if config["mission"].startswith("task"))
    opponent = next(config for config in config_aircraft.values() if not config["mission"].startswith("task"))
    aircraft_list = ([("B", dict(rl_agent), None) for _ in range(NUM_RL_AGENTS)]
                     + [("B", {**rl_agent, "mission": "model_vrf"}, marking) for marking in VRF_TEAMMATES]
                     + [("R", dict(opponent), None) for _ in range(NUM_SIM_OPPONENTS)]
                     + [("R", {**opponent, "mission": "model_vrf"}, marking) for marking in VRF_OPPONENTS])
    aircraft = {aid: None for aid in config_aircraft}
    vrf_entities = {}
    for number, (team, config, marking) in enumerate(aircraft_list, start=1):
        aid = f"{team}{number}"
        aircraft[aid] = config
        if config["mission"] == "model_vrf":
            vrf_entities[aid] = marking
    return aircraft, vrf_entities


def print_states(env):
    """Prints the states of the RL agents with their closest opponents, and the DIS traffic."""
    states = []
    for aid in env.rl_agent_ids:
        agent = env.agents[aid]
        if not agent.alive:
            states.append(f"{aid}: {'shot down' if agent.shotdown else 'crashed'}")
            continue
        state = f"{aid}: {agent.get_altitude():.0f} m, {np.linalg.norm(agent.get_velocity()):.0f} m/s"
        closest = agent.mission.get_closest_opp(agent)
        if closest:
            opponent, distance = closest[0]
            state += f", {distance / 1000:.1f} km to {env.dis.label(opponent.aid)}"
        states.append(state)
    print(f"{env.dis.time:6.1f} s | " + " | ".join(states) + f" | {env.dis.traffic()}")


def run_inference():
    if VRF_SIMULATION:
        warnings.warn("VRF_SIMULATION = True (all aircraft simulated by VR-F, only the actions computed here) is not "
                      "implemented yet, set it to False.")
        return
    if LIST_ENTITIES:
        from comm.dis_comm import list_entities
        list_entities(address=DIS_ADDRESS, port=DIS_PORT, exercise_id=EXERCISE_ID, site_id=SITE_ID,
                      application_id=APPLICATION_ID)
        return

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

    num_team_b = NUM_RL_AGENTS + len(VRF_TEAMMATES)
    num_team_r = NUM_SIM_OPPONENTS + len(VRF_OPPONENTS)
    assert NUM_RL_AGENTS > 0 and num_team_r > 0, "Set at least one RL agent and one simulated or VR-F opponent"
    aircraft, vrf_entities = roster(args.task)
    config_overrides = {"battle_field_center": [LOCATION[0], LOCATION[1], 0.0], "aircraft": aircraft,
                        "num_rl_agents": num_team_b, "num_opp_agents": num_team_r}
    if HORIZON is not None:
        config_overrides["horizon"] = HORIZON
    kwargs = {"hierarchy_level": args.hierarchy_level, "training_level": args.level+int(args.task == "track" or args.task == "escape" or args.hierarchy_level == 2),
              "rendering": False, "config_overrides": config_overrides, "dis": dis_options(vrf_entities)}

    env = AircombatMARL(task=args.task, **kwargs)
    ep_rewards, ep_lengths = [], []
    wins = losses = 0
    try:
        actors, _ = build_actor_critic(
            args, env.get_obs_size(), env.get_action_space(),
            env.get_state_size(), env.n_agents, device,
        )
        actors.load_state_dict(load_checkpoint(args.restore_path, device)["actors_state_dict"])
        actors.eval()

        agent_keys = env.agent_keys
        A = env.n_agents
        L, H = args.rnn_num_layers, args.actor_hidden_dim
        reset_options = {"init_states": INIT_STATES} if INIT_STATES else None

        for ep in range(NUM_EPISODES):
            obs_dict, _ = env.reset(seed=args.seed + ep, options=reset_options)
            obs = np.array([obs_dict[k] for k in agent_keys], dtype=np.float32)

            rnn_hxs = torch.zeros(1, A, L, H, device=device) if args.rnn else None
            masks = torch.ones(1, device=device) if args.rnn else None

            ep_reward, ep_len, done = 0.0, 0, False
            last_status = time.perf_counter()
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

                if SHOW_STATES and time.perf_counter() - last_status >= STATUS_INTERVAL:
                    last_status = time.perf_counter()
                    print_states(env.env)

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
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        env.close()  # removes the aircraft of this env from VR-F

    if ep_rewards:
        n = len(ep_rewards)
        print(f"\nMean reward: {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}  "
              f"| Mean length: {np.mean(ep_lengths):.1f}")
        print(f"Mean Wins: {wins/n:.3f} | Mean Losses: {losses/n:.3f}")


if __name__ == "__main__":
    run_inference()
