import math
import numpy as np
from rewards.reward_base import BaseRewardFunction


# class RollPitch(BaseRewardFunction):
#     """
#     Punish abrupt changes in roll and pitch.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.prev_roll = {}
#         self.prev_pitch = {}
#         self.max_delta_roll = 8
#         self.max_delta_pitch = 4

#     def get_reward(self, task, env, agent_id, action=None):
#         return self._process(self.roll_pitch_punish(env, agent_id, action), agent_id)
    
#     def reset(self, task, env):
#         self.prev_pitch.clear()
#         self.prev_roll.clear()
#         super().reset(task, env)

#     def roll_pitch_punish(self, env, agent_id, action):
#         rew = 0
#         current_roll = action[agent_id][0]
#         current_pitch = action[agent_id][1]

#         if env.steps <= 1 or agent_id not in self.prev_roll:
#             self.prev_roll[agent_id] = current_roll
#             self.prev_pitch[agent_id] = current_pitch
#             return rew
        
#         if abs(current_roll - self.prev_roll[agent_id]) > self.max_delta_roll:
#             rew -= 0.1

#         if abs(current_pitch - self.prev_pitch[agent_id]) > self.max_delta_pitch:
#             rew -= 0.1

#         self.prev_roll[agent_id] = current_roll
#         self.prev_pitch[agent_id] = current_pitch

#         return rew



# class RollPitch(BaseRewardFunction):
#     """
#     Smooth penalty on aileron/elevator action rate and acceleration.

#     rate term:  k1 * (delta / N)^2          — penalizes any change proportionally
#     accel term: k2 * (accel / (2*N))^2      — penalizes direction reversals (oscillation)

#     Both terms are smooth everywhere, so the gradient exists for mild oscillations
#     that a binary threshold would miss entirely.
#     """
#     AILERON_BINS = 20   # max bin index for aileron  (21 bins -> 0..20)
#     ELEVATOR_BINS = 10  # max bin index for elevator (11 bins -> 0..10)

#     def __init__(self, config):
#         super().__init__(config)
#         self.k1 = 0.05   # rate coefficient
#         self.k2 = 0.15   # acceleration coefficient (oscillation-sensitive)
#         self.prev_action = {}   # {agent_id: [aileron_idx, elevator_idx]}
#         self.prev_delta  = {}   # {agent_id: [d_aileron,  d_elevator]}

#     def reset(self, task, env):
#         self.prev_action.clear()
#         self.prev_delta.clear()
#         super().reset(task, env)

#     def get_reward(self, task, env, agent_id, action=None):
#         return self._process(self._smooth_penalty(env, agent_id, action), agent_id)

#     def _smooth_penalty(self, env, agent_id, action):
#         cur = np.array(action[agent_id][:2], dtype=float)  # [aileron_idx, elevator_idx]
#         N   = np.array([self.AILERON_BINS, self.ELEVATOR_BINS], dtype=float)

#         if env.steps <= 1 or agent_id not in self.prev_action:
#             self.prev_action[agent_id] = cur
#             self.prev_delta[agent_id]  = np.zeros(2)
#             return 0.0

#         delta = cur - self.prev_action[agent_id]           # first derivative
#         accel = delta - self.prev_delta[agent_id]          # second derivative

#         rate_penalty  = self.k1 * np.sum((delta / N) ** 2)
#         accel_penalty = self.k2 * np.sum((accel / (2.0 * N)) ** 2)

#         self.prev_action[agent_id] = cur
#         self.prev_delta[agent_id]  = delta

#         return -(rate_penalty + accel_penalty)


class RollPitch(BaseRewardFunction):
    """
    Smooth penalty on aileron/elevator command rate and command jerk.

    The environment passes raw discrete action indices into rewards, so this
    first maps the roll/pitch action bins back to their absolute command values.
    """
    def __init__(self, config):
        super().__init__(config)
        self.roll_min = -0.8
        self.roll_max = 0.8
        self.pitch_min = -0.8
        self.pitch_max = 0.8

        self.rate_coef = np.array([0.03, 0.015], dtype=np.float64)
        self.jerk_coef = np.array([0.08, 0.04], dtype=np.float64)

        self.prev_cmd = {}
        self.prev_delta = {}

    def reset(self, task, env):
        self.prev_cmd.clear()
        self.prev_delta.clear()
        super().reset(task, env)

    def get_reward(self, task, env, agent_id, action=None):
        return self._process(self._smooth_penalty(task, env, agent_id, action), agent_id)

    def _smooth_penalty(self, task, env, agent_id, action):
        if action is None or agent_id not in action:
            return 0.0

        current_cmd = self._scaled_roll_pitch_cmd(task, action[agent_id])
        if env.steps <= 1 or agent_id not in self.prev_cmd:
            self.prev_cmd[agent_id] = current_cmd
            self.prev_delta[agent_id] = np.zeros(2, dtype=np.float64)
            return 0.0

        delta = current_cmd - self.prev_cmd[agent_id]
        jerk = delta - self.prev_delta[agent_id]

        rate_penalty = np.sum(self.rate_coef * np.square(delta))
        jerk_penalty = np.sum(self.jerk_coef * np.square(jerk))

        self.prev_cmd[agent_id] = current_cmd
        self.prev_delta[agent_id] = delta

        return -float(rate_penalty + jerk_penalty)

    def _scaled_roll_pitch_cmd(self, task, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float64)
        nvec = np.asarray(task.action_space.nvec, dtype=np.float64)
        roll = self._scale_action(raw_action[0], nvec[0], self.roll_min, self.roll_max)
        pitch = self._scale_action(raw_action[1], nvec[1], self.pitch_min, self.pitch_max)
        return np.array([roll, pitch], dtype=np.float64)

    @staticmethod
    def _scale_action(action_value, num_bins, low, high):
        if num_bins <= 1:
            return float(low)
        return float(action_value * (high - low) / (num_bins - 1) + low)
