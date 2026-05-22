from rewards.reward_base import BaseRewardFunction
import math


class DistanceReward(BaseRewardFunction):
    """
    Higher reward the further away from opponent. 
    Designed as potential reward.
    """
    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50) * 1000 # map size in m

    def get_reward(self, task, env, agent_id, action=None):
        # _, _, a_r, a_rr, a_pr, dist = super().get_orientation_metrics(env, agent_id, two_d=True)
        # dist_r = dist / math.sqrt(self.map_size**2 + self.map_size**2)
        # roll_scale = 30
        # roll_r = math.exp(-((a_r/roll_scale)**2))
        # reward = dist_r * roll_r

        reward = self.dist_reward(env, agent_id)
        return self._process(reward, agent_id)



    def dist_reward(self, env, agent_id):

        def delta_loc(x):
            # in [KM] 
            dist_scale = 7
            dist_offset = 5
            if x < dist_offset:
                dist_r = 0
            else:
                dist_r = 1-math.exp(-(((x-dist_offset)/dist_scale)**2))
            return dist_r
        
        _, a_t, _, _, _, _ = super().get_orientation_metrics(env, agent_id, deg=True)
        delta_x, delta_y, _, _, _, _ = super().get_delta_values(env, agent_id, in_km=True)
        d = 0.5 * delta_loc(delta_x) + 0.5 * delta_loc(delta_y)

        head_scale = 50
        head_t = math.exp(-((a_t/head_scale)**2))

        reward = 0.7*d + 0.3*head_t
        return reward