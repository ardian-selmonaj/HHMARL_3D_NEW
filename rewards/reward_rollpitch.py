import math
import numpy as np
from rewards.reward_base import BaseRewardFunction


class RollPitch(BaseRewardFunction):
    """
    Punish abrupt changes in roll and pitch.
    """
    def __init__(self, config):
        super().__init__(config)
        self.max_delta = getattr(self.config, f'{self.__class__.__name__}_max_delta', 15)
        self.prev_roll = {}
        self.prev_pitch = {}

    def get_reward(self, task, env, agent_id, action=None):
        return self.roll_pitch_punish(env, agent_id, action)

    def roll_pitch_punish(self, env, agent_id, action):
        rew = 0
        current_roll = action[agent_id][0]
        current_pitch = action[agent_id][1]

        if env.steps <= 1 or agent_id not in self.prev_roll:
            self.prev_roll[agent_id] = current_roll
            self.prev_pitch[agent_id] = current_pitch
            return rew
        
        if abs(current_roll - self.prev_roll[agent_id]) > self.max_delta:
            rew -= 0.1

        if abs(current_pitch - self.prev_pitch[agent_id]) > self.max_delta:
            rew -= 0.1

        self.prev_roll[agent_id] = current_roll
        self.prev_pitch[agent_id] = current_pitch

        return rew