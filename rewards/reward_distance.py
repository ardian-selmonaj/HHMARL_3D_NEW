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
        _, _, a_r, a_rr, a_pr, dist = super().get_orientation_metrics(env, agent_id, two_d=True)
        dist_r = dist / math.sqrt(self.map_size**2 + self.map_size**2)
        roll_scale = 30
        roll_r = math.exp(-((a_r/roll_scale)**2))
        reward = dist_r * roll_r
        return self._process(reward, agent_id)
