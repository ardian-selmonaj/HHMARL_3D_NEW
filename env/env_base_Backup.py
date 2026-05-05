import logging
import importlib
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from env.sim import AircraftSim
from env.utils import parse_config
from rendering.scene import Scene

class AircraftEnv(gym.Env):
    """
    Base Class for Aircraft Combat.
    Without overwriting, this class is used for low-level policy training as is.
    """
    #future rendering choices.
    metadata = {"render.modes": ["simple", "fg"]}

    def __init__(self, config: dict):
        """
        Loading env parameters. 
            - config contains environment specific parameters
            - self.config additionally contains mission based parameters, saved in .yaml files
        """
        self.log = logging.getLogger(__name__)
        self.config = parse_config(config["mission_config"])
        self.hierarchy_level = config["hierarchy_level"]
        self.training_level = config['training_level']
        self.rnd_init_states = getattr(self.config, 'rnd_init', True)
        self.horizon = getattr(self.config, 'horizon', 1000)
        self.sim_freq = getattr(self.config, 'sim_freq', 100)
        self.sim_steps = getattr(self.config, 'sim_steps', 10)
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (0.0, 0.0, 0.0))
        self.format = getattr(self.config, 'space_format', "ray")
        self.steps = 0
        self.load()

    @property
    def time_interval(self) -> int:
        return self.sim_steps / self.sim_freq

    def load(self):
        """
        Load format of training algorithm, missions, agent simulators and spaces.
        """
        self.algo_format = getattr(self, f"{self.format}_format", None)
        assert self.algo_format, self.log.error("Specify an Algorithm format.")

        self.load_missions()
        self.load_agents()
        self.load_spaces()
        self.seed()
        self.scene = None

    def load_missions(self):
        """
        Load all possible missions (tasks and models) that can be assigned to each agent.
        """
        self.missions={}
        for config in self.config.aircraft.values():
            mission = config.get("mission", None)
            assert mission, self.log.error("Specify mission per aircraft in .yaml config.")
            mission_type, mission_name = mission.split("_")
            try:
                miss = getattr(importlib.import_module("missions"), 
                                                      f"{mission_name.capitalize()}{mission_type.capitalize()}")
                if not mission in self.missions.keys():
                    self.missions[mission] = miss(self.config)
            except:
                raise NotImplementedError(f'Unknown Mission: {mission_name.capitalize()}{mission_type.capitalize()} or error in instance construction.')

    def load_agents(self):
        """
        Load all JSBSim aircraft.
        """
        self.agents = {}
        self.teams = {} #{team:aid}
        self.rl_agent_ids = []
        self.opp_agent_ids = []
        for aid, config in self.config.aircraft.items():
            mission = config.get("mission", None)
            mission_type = mission.split("_")[0]
            self.agents[aid] = AircraftSim(
                tid=aid[0],
                aid=aid,
                mission=self.missions[mission],
                learning=mission_type=="task",
                model=config.get("model", "f16"),
                num_missiles=config.get("num_missiles", 5),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (0.0, 0.0, 0.0)),
                sim_freq=self.sim_freq,
                sim_steps=self.sim_steps,
                unlimited_fuel=config.get("unlimited_fuel", True))

            if mission_type=="task":
                self.rl_agent_ids.append(aid)
            else:
                self.opp_agent_ids.append(aid)

            try:
                self.teams[aid[0]].append(aid)
            except:
                self.teams[aid[0]] = [aid]
            
        for aid, agent in self.agents.items():
            for tid, team in self.teams.items():
                if agent.tid == tid:
                    aid_team = team.copy()
                    aid_team.remove(aid)
                    agent.partners = [self.agents[a] for a in aid_team]
                else:
                    agent.opponents = [self.agents[a] for a in team]
        
    def load_spaces(self) -> gym.Space:
        """
        Load and format observation and action spaces according to algorithm.
        """
        obs = {}
        act = {}
        for aid, agent in self.agents.items():
            if agent.is_learning:
                obs.update(self.algo_format(data=agent.mission.observation_space, aid=aid, type="space"))
                act.update(self.algo_format(data=agent.mission.action_space, aid=aid, type="space"))
        if self.format=="ray":
            self.observation_space= gym.spaces.Dict(obs)
            self.action_space= gym.spaces.Dict(act)
        else:
            self.observation_space = obs
            self.action_space = act
        self.log.info(f"Obseravtion space={self.observation_space}, Action space={self.action_space} \n")

    def reset_agent_states(self):
        """
        Assigning init conditions.
        """
        def simple_init_attitude(attitude_type, agent):
            if attitude_type == "head":
                if np.random.rand() < 0.5:
                    return np.random.uniform(0, 30)
                else:
                    return np.random.uniform(330, 360)
            elif attitude_type == "altitude":
                return np.random.randint(25000, 26000)
            elif attitude_type == "velocity":
                if agent.tid == "B":
                    return np.random.randint(700,900)
                else:
                    return np.random.randint(500,700)
            
        def simple_init_location(loc, agent):
            if self.training_level == 1:
                if loc == "long":
                    return np.random.uniform(-0.02, 0.02)
                if agent.tid == "B":
                    return np.random.uniform(-0.06, -0.01)
                else:
                    return np.random.uniform(0.01, 0.06)
            else:
                return np.random.uniform(-0.13, 0.13)

        init_states = [agent.init_state.copy() for agent in self.agents.values()]

        for idx, agent in enumerate(self.agents.values()):
            if self.rnd_init_states:
                for sk in init_states[idx].keys():
                    if sk == "ic_psi_true_deg":
                        init_states[idx][sk] = simple_init_attitude("head", agent) if self.training_level == 1 else np.random.randint(0,350)
                    elif sk in ["ic_long_gc_deg", "ic_lat_geod_deg"]:
                        init_states[idx][sk] = np.random.uniform(-0.15, 0.15) if self.hierarchy_level == 1 else simple_init_location(sk.split("_")[1], agent)
                    elif sk == "ic_h_sl_ft":
                        init_states[idx][sk] = simple_init_attitude("altitude", agent) if self.training_level == 1 else np.random.randint(22000, 28000)
                    elif sk == "ic_u_fps":
                        init_states[idx][sk] = simple_init_attitude("velocity", agent) if self.training_level == 1 else np.random.randint(500,1000)
                    else:
                        raise ValueError(f"Init state key {sk} not randomized.")
            agent.reload(new_state=init_states[idx])

    def reset(self) -> np.ndarray:
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.steps = 0
        self.load_agents()
        self.reset_agent_states()
        for mission in self.missions.values():
            mission.reset(self)
        return self.get_obs(), {}

    def step(self, action: Any) -> Tuple[dict, dict, dict, dict]:
        """
        Run one timestep of the environment's dynamics.
        """
        self.steps += 1
        rewards = {}
        obs = {}
        terminated = {}
        truncated = {}

        #run simulation
        for it in range(self.sim_steps+3):
            for aid, agent in self.agents.items():
                if it == 0:
                    act = action.get(aid, None)
                    act = agent.mission.get_action(action=act, agent=agent, env=self)

                    #### REL ACT
                    if not agent.is_learning and self.training_level <= 2:
                        agent.set_property_values(agent.mission.action_var, act)
                    else:
                        act_values = agent.get_property_values(agent.mission.action_var)
                        for ka, act_val in enumerate(act_values):
                            act[ka] += act_val
                        agent.set_property_values(agent.mission.action_var, act)

                elif it == self.sim_steps+1:
                    obs.update(self.get_obs(aid))
                    if agent.is_learning:
                        terminated[aid] = bool(agent.mission.get_termination(self, aid))
                        truncated[aid] = self.steps >= self.horizon
                elif it == self.sim_steps+2:
                    rewards.update(self.get_reward(aid, action))
                    agent.mission.step(self, agent)
                else:
                    agent.run()

        terminated["all"] = any(terminated[aid] for aid in self.rl_agent_ids)
        truncated["all"] = self.steps >= self.horizon
        self.log.debug(f"Method step(), obs={obs}, term={terminated}, trunc={truncated} \n")
        return obs, rewards, terminated, truncated, {}

    def get_obs(self, aid:str=None):
        """
        Returns all learning agent observations in a list.
        """
        obs = {}
        if aid:
            agent = self.agents[aid]
            if agent.is_learning:
                obs.update(self.algo_format(data=agent.mission.get_obs(self, agent), aid=aid))
        else:
            for aid, agent in self.agents.items():
                if agent.is_learning:
                    obs.update(self.algo_format(data=agent.mission.get_obs(self, agent), aid=aid))

        for k, v in obs.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    assert not np.isnan(sub_v).any(), self.log.error(f"Observation of agent {k}.{sub_k} contains NaN values: {sub_v}")
            else:
                assert not np.isnan(v).any(), self.log.error(f"Observation of agent {k} contains NaN values: {v}")
        return obs

    def get_reward(self, aid:str, action:dict, raw_value:bool = False):
        if self.agents[aid].is_learning:
            if raw_value:
                return self.agents[aid].mission.get_reward(self, aid, action)
            else:
                return self.algo_format(data=self.agents[aid].mission.get_reward(self, aid, action), aid=aid, type="rew")
        else:
            return {}

    def close(self):
        """
        Cleans up this environment's objects
        """
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()

    def render(self, viewpoint=0, trajectory_view=False):
        """
        Render the scene.
        """
        if not self.scene:
            self.scene = Scene(self)
        self.scene.render(viewpoint, trajectory_view)
        return

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def ray_format(self, data:Any, aid:str, type:str="data") -> Dict[str, np.array]:
        """
        Format observation and action spaces according to Ray RLlib.
        """
        # out = {aid:np.array(data, dtype=np.float32) if type=="data" else data}
        out = {aid: data}
        #self.log.debug(f"Method ray_format output: out={out}, type={type} \n")
        return out
    
    def pymarl_format(self):
        """
        Format observation and action spaces according to PyMARL.
        """
        raise NotImplementedError("PyMarl Format not specified.")
    
    def custom_format(self):
        """
        Custom format observation and action spaces.
        """
        raise NotImplementedError("Custom Format not specified.")


class AircraftEnvHier(AircraftEnv):
    """
    AirCombatEnv for Hierarchcal Training.
    """
    def __init__(self, config: dict, inference:bool = False, rendering:bool = False, render_view:int = 0):
        super().__init__(config)
        self.hl_sim_steps = getattr(self.config, 'hl_sim_steps', 15)
        self.inference = inference
        self.rendering = rendering
        self.render_view = render_view
        self.dead_agent_ids = [] #required for rendering
        assert len(self.rl_agent_ids) > 0 and len(self.opp_agent_ids) > 0, f"RL agents {len(self.rl_agent_ids)} and Opp agents {len(self.opp_agent_ids)}"

    def step(self, action):
        """
        Execute high-level actions.
        """
        self.steps += 1
        rewards = {aid: 0 for aid, agent in self.agents.items() if agent.is_learning}
        trunc = term = {aid: False for aid, agent in self.agents.items() if agent.is_learning}
        dones = {aid: False for aid in self.agents.keys()}
        obs = {}

        for aid, agent in self.agents.items():
            act = action.get(aid, None)
            agent.mission.set_policy(action=act, agent=agent, env=self)

        termination_condition = False
        i = 0
        while i<=self.hl_sim_steps and not termination_condition:
            if self.rendering: self.render(viewpoint=self.render_view)
            
            for aid, agent in self.agents.items():
                agent.mission.run_action(agent=agent, env=self)
                agent_termination = agent.mission.get_termination(self, aid)
                if agent_termination:
                    termination_condition = True
                    dones[aid] = True
                    if agent.is_learning:
                        term[aid] = True
                if agent.is_learning:
                    rewards[aid] += self.get_reward(aid, action, raw_value=True)
                agent.mission.step(self, agent)
            i+= 1

        if self.steps >= self.horizon:
            for aid in trunc.keys():
                # mark as truncated if not already terminated
                if not term[aid]:
                    trunc[aid] = True

        # during training, need to provide observation also for dead agents because of TD bootstrapping
        if not self.inference:
            for aid in self.agents.keys():
                obs.update(self.get_obs(aid))

        for aid, dead in dones.items():
            dead_agent = None
            if dead: 
                dead_agent = self.agents.get(aid, None)
            if dead_agent:
                for agent in self.agents.values():
                    if agent.aid != dead_agent.aid:
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

        # otherwise, during testing, can provide updated observation directly
        if self.inference:
            for aid in self.agents.keys():
                obs.update(self.get_obs(aid))

        all_opps_dead, learn, non_learn = self.check_dead_agents()

        term['__all__'] = trunc['__all__'] = len(self.rl_agent_ids)==0 or len(self.opp_agent_ids) == 0 or all_opps_dead or learn == 0 or non_learn == 0 or self.steps >= self.horizon

        if self.inference:
            return obs, rewards, term, trunc, {"dones": dones}
        else:
            return obs, rewards, term, trunc, {}
    
    def check_dead_agents(self):
        learning = 0
        non_learning = 0
        for agent in self.agents.values():
            if agent.is_learning: learning += 1
            if not agent.is_learning: non_learning += 1
        all_opps_dead = False
        for agent in self.agents.values():
            if len(agent.opponents) == 0:
                all_opps_dead = True
        return all_opps_dead, learning, non_learning