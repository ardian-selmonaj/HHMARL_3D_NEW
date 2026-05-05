import logging
import importlib
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from typing import Any, Tuple
from env.sim import AircraftSim
from env.utils import parse_config
from rendering.scene import Scene


class AircraftEnv(gym.Env):
    """Base class for low-level aircraft combat training."""

    metadata = {"render.modes": ["simple", "fg"]}

    def __init__(self, config: dict):
        self.log = logging.getLogger(__name__)
        self.config = parse_config(config["mission_config"])
        self.hierarchy_level = config["hierarchy_level"]
        self.training_level = config["training_level"]
        self.rnd_init_states = getattr(self.config, "rnd_init", True)
        self.horizon = getattr(self.config, "horizon", 1000)
        self.sim_freq = getattr(self.config, "sim_freq", 100)
        self.sim_steps = getattr(self.config, "sim_steps", 10)
        self.center_lon, self.center_lat, self.center_alt = getattr(
            self.config, "battle_field_center", (0.0, 0.0, 0.0)
        )
        self.steps = 0
        self.scene = None
        self._alive_last_step = set()
        self.load_missions()
        self.load_agents()
        self.load_spaces()
        self.seed()

    def load_missions(self):
        self.missions = {}
        for config in self.config.aircraft.values():
            mission = config.get("mission")
            assert mission, "Specify mission per aircraft in .yaml config."
            if mission not in self.missions:
                mission_type, mission_name = mission.split("_")[0], mission.split("_")[1]
                class_name = f"{mission_name.capitalize()}{mission_type.capitalize()}"
                try:
                    cls = getattr(importlib.import_module("missions"), class_name)
                except AttributeError as e:
                    raise NotImplementedError(f"Unknown Mission: {class_name}") from e
                self.missions[mission] = cls(self.config)

    def load_agents(self):
        self.agents = {}
        self.teams = {}
        self.rl_agent_ids = []
        self.opp_agent_ids = []
        origin = getattr(self.config, "battle_field_center", (0.0, 0.0, 0.0))
        for aid, config in self.config.aircraft.items():
            mission = config.get("mission")
            is_learning = mission.split("_")[0] == "task"
            self.agents[aid] = AircraftSim(
                tid=aid[0],
                aid=aid,
                mission=self.missions[mission],
                learning=is_learning,
                model=config.get("model", "f16"),
                num_missiles=config.get("num_missiles", 5),
                init_state=config.get("init_state"),
                origin=origin,
                sim_freq=self.sim_freq,
                sim_steps=self.sim_steps,
                unlimited_fuel=config.get("unlimited_fuel", True),
            )
            (self.rl_agent_ids if is_learning else self.opp_agent_ids).append(aid)
            self.teams.setdefault(aid[0], []).append(aid)

        for aid, agent in self.agents.items():
            agent.partners = [self.agents[a] for a in self.teams[agent.tid] if a != aid]
            agent.opponents = [
                self.agents[a]
                for tid, team in self.teams.items() if tid != agent.tid
                for a in team
            ]

    def load_spaces(self):
        obs, act = {}, {}
        for aid, agent in self.agents.items():
            if agent.is_learning:
                obs[aid] = agent.mission.observation_space
                act[aid] = agent.mission.action_space
        self.observation_space = gym.spaces.Dict(obs)
        self.action_space = gym.spaces.Dict(act)
        self.log.info(f"Observation space={self.observation_space}, Action space={self.action_space}")

    def reset_agent_states(self):
        lvl1 = self.training_level == 1
        hier1 = self.hierarchy_level == 1
        rng = self.np_random
        for agent in self.agents.values():
            state = agent.init_state.copy()
            if self.rnd_init_states:
                is_B = agent.tid == "B"
                for sk in state:
                    if sk == "ic_psi_true_deg":
                        state[sk] = (
                            (rng.uniform(0, 30) if rng.random() < 0.5 else rng.uniform(330, 360))
                            if lvl1 else int(rng.integers(0, 350))
                        )
                    elif sk in ("ic_long_gc_deg", "ic_lat_geod_deg"):
                        if hier1:
                            state[sk] = rng.uniform(-0.15, 0.15)
                        elif lvl1:
                            if sk == "ic_long_gc_deg":
                                state[sk] = rng.uniform(-0.02, 0.02)
                            elif is_B:
                                state[sk] = rng.uniform(-0.06, -0.01)
                            else:
                                state[sk] = rng.uniform(0.01, 0.06)
                        else:
                            state[sk] = rng.uniform(-0.13, 0.13)
                    elif sk == "ic_h_sl_ft":
                        state[sk] = int(rng.integers(25000, 26000)) if lvl1 else int(rng.integers(22000, 28000))
                    elif sk == "ic_u_fps":
                        state[sk] = (
                            int(rng.integers(700, 900)) if (lvl1 and is_B)
                            else int(rng.integers(500, 700)) if lvl1
                            else int(rng.integers(500, 1000))
                        )
                    else:
                        raise ValueError(f"Init state key {sk} not randomized.")
            agent.reload(new_state=state)

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.reset_agent_states()
        for mission in self.missions.values():
            mission.reset(self)
        self._alive_last_step = {aid for aid in self.agents}
        return self.get_obs(), {}

    def step(self, action: Any) -> Tuple[dict, dict, dict, dict, dict]:
        self.steps += 1
        timed_out = self.steps >= self.horizon

        # Phase 0: apply actions
        for aid, agent in self.agents.items():
            if not agent.alive:
                continue

            act = agent.mission.get_action(action=action.get(aid), agent=agent, env=self)
            agent.set_property_values(agent.mission.action_var, act)
            
            # if not agent.is_learning and self.training_level <= 2:
            #     agent.set_property_values(mission.action_var, act)
            # else:
            #     act_vals = agent.get_property_values(mission.action_var)
            #     for k, v in enumerate(act_vals):
            #         act[k] += v
            #     agent.set_property_values(mission.action_var, act)

        # Phase 1: physics steps
        for _ in range(self.sim_steps):
            for agent in self.agents.values():
                agent.run()

        # Phase 2: observations and terminations
        obs = {}
        terminated = {}
        for aid, agent in self.agents.items():
            obs.update(self.get_obs(aid))
            if agent.is_learning:
                terminated[aid] = bool(agent.mission.get_termination(self, aid))

        # Phase 3: rewards and mission step
        rewards = {}
        for aid, agent in self.agents.items():
            rewards.update(self.get_reward(aid, action))
            agent.mission.step(self, agent)

        info = self._combat_info()
        self._alive_last_step = {aid for aid, agent in self.agents.items() if agent.alive}
        terminated["all"] = all(terminated[aid] for aid in self.rl_agent_ids) or self._opponents_eliminated()
        truncated = {aid: timed_out for aid in self.rl_agent_ids}
        truncated["all"] = timed_out
        self.log.debug(f"step={self.steps}, term={terminated}, trunc={truncated}")
        return obs, rewards, terminated, truncated, info

    def get_obs(self, aid: str = None) -> dict:
        if aid is not None:
            agent = self.agents[aid]
            obs = {aid: agent.mission.get_obs(self, agent)} if agent.is_learning else {}
        else:
            obs = {
                aid: agent.mission.get_obs(self, agent)
                for aid, agent in self.agents.items() if agent.is_learning
            }
        for k, v in obs.items():
            assert not np.isnan(v).any(), self.log.error(f"Observation of agent {k} contains NaN: {v}")
        return obs

    def get_reward(self, aid: str, action: dict, raw_value: bool = False) -> dict:
        if not self.agents[aid].is_learning:
            return {}
        rew = self.agents[aid].mission.get_reward(self, aid, action)
        return rew if raw_value else {aid: rew}

    def _opponents_eliminated(self) -> bool:
        return any(
            all(not self.agents[aid].alive for aid in team_aids)
            for team_aids in self.teams.values()
        )

    def _combat_info(self) -> dict:
        opponents = [self.agents[aid] for aid in self.opp_agent_ids]
        friendlies = [self.agents[aid] for aid in self.rl_agent_ids]
        return {
            "opp_killed":   any(a.shotdown and a.aid in self._alive_last_step for a in opponents),
            "opp_crashed":  any(a.crashed  and a.aid in self._alive_last_step for a in opponents),
            "agent_killed": any(a.shotdown and a.aid in self._alive_last_step for a in friendlies),
            "agent_crashed":any(a.crashed  and a.aid in self._alive_last_step for a in friendlies),
        }

    def close(self):
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()

    def render(self, viewpoint=0, trajectory_view=False):
        if not self.scene:
            self.scene = Scene(self)
        self.scene.render(viewpoint, trajectory_view)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class AircraftEnvHier(AircraftEnv):
    """AircraftEnv for hierarchical training."""

    def __init__(self, config: dict, inference: bool = False, rendering: bool = False, render_view: int = 0):
        super().__init__(config)
        self.hl_sim_steps = getattr(self.config, "hl_sim_steps", 15)
        self.inference = inference
        self.rendering = rendering
        self.render_view = render_view
        self.dead_agent_ids = []
        assert len(self.rl_agent_ids) > 0 and len(self.opp_agent_ids) > 0, \
            f"RL agents {len(self.rl_agent_ids)} and Opp agents {len(self.opp_agent_ids)}"

    def step(self, action):
        self.steps += 1
        rewards = {aid: 0 for aid, agent in self.agents.items() if agent.is_learning}
        trunc = term = {aid: False for aid, agent in self.agents.items() if agent.is_learning}
        dones = {aid: False for aid in self.agents.keys()}
        obs = {}

        for aid, agent in self.agents.items():
            agent.mission.set_policy(action=action.get(aid, None), agent=agent, env=self)

        termination_condition = False
        i = 0
        while i <= self.hl_sim_steps and not termination_condition:
            if self.rendering:
                self.render(viewpoint=self.render_view)
            for aid, agent in self.agents.items():
                agent.mission.run_action(agent=agent, env=self)
                if agent.mission.get_termination(self, aid):
                    termination_condition = True
                    dones[aid] = True
                    if agent.is_learning:
                        term[aid] = True
                if agent.is_learning:
                    rewards[aid] += self.get_reward(aid, action, raw_value=True)
                agent.mission.step(self, agent)
            i += 1

        if self.steps >= self.horizon:
            for aid in trunc:
                if not term[aid]:
                    trunc[aid] = True

        if not self.inference:
            for aid in self.agents:
                obs.update(self.get_obs(aid))

        for aid, dead in dones.items():
            if not dead:
                continue
            dead_agent = self.agents.get(aid)
            if not dead_agent:
                continue
            for agent in self.agents.values():
                if agent.aid == dead_agent.aid:
                    continue
                if agent.tid == dead_agent.tid:
                    agent.partners.remove(dead_agent)
                else:
                    agent.opponents.remove(dead_agent)
            if dead_agent.aid in self.rl_agent_ids:
                self.rl_agent_ids.remove(dead_agent.aid)
            elif dead_agent.aid in self.opp_agent_ids:
                self.opp_agent_ids.remove(dead_agent.aid)
            self.dead_agent_ids.append(dead_agent.aid)
            del self.agents[dead_agent.aid]

        if self.inference:
            for aid in self.agents:
                obs.update(self.get_obs(aid))

        all_opps_dead, learn, non_learn = self.check_dead_agents()
        all_done = (
            len(self.rl_agent_ids) == 0 or len(self.opp_agent_ids) == 0
            or all_opps_dead or learn == 0 or non_learn == 0
            or self.steps >= self.horizon
        )
        term["__all__"] = trunc["__all__"] = all_done

        if self.inference:
            return obs, rewards, term, trunc, {"dones": dones}
        return obs, rewards, term, trunc, {}

    def check_dead_agents(self):
        learning = sum(1 for a in self.agents.values() if a.is_learning)
        non_learning = sum(1 for a in self.agents.values() if not a.is_learning)
        all_opps_dead = any(len(a.opponents) == 0 for a in self.agents.values())
        return all_opps_dead, learning, non_learning
