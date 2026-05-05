import numpy as np
from missions.mission_base import BaseMission


class StaticModel(BaseMission):
    """
    Rule-based agent to fly straight.
    """
    def __init__(self, config):
        super().__init__(config)

    def load_obs_space(self):
        pass

    def load_obs_vars(self):
        pass

    def load_rewards(self):
        self.rewards = []

    def get_obs(self, env, agent, **kwargs):
        pass

    def get_action(self, action=None, agent=None):
        action = np.array([20, 18.6, 20, 0])
        return super().self.get_action(action)

    def reset(self):
        pass

    