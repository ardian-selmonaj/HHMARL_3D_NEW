from env.env_base import AircraftEnv, AircraftEnvHier
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np

class AircraftRay(MultiAgentEnv):
    """
    A wrapped AircraftEnv for Ray RLlib.
    """
    def __init__(self, config:dict):
        #self._skip_env_checking = True
        if config["hierarchy_level"] == 0:
            self.env = AircraftEnv(config)
        else:
            self.env = AircraftEnvHier(config)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.agents = self.possible_agents = self.env.rl_agent_ids

        super().__init__()

    def reset(self, *, seed=None, options=None):
        return self.env.reset()
    
    def step(self, actions):
        return self.env.step(actions)
        
    def render(self):
        self.env.render()