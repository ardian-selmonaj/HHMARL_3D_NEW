from rewards.reward_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile or cannon: -5
    - Crash accidentally: -5
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id, action=None):
        reward = 0
        if env.agents[agent_id].shotdown:
            reward -= 5 #10
        elif env.agents[agent_id].crashed:
            reward -= 5 #10
        return self._process(reward, agent_id)
