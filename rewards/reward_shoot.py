from rewards.reward_base import BaseRewardFunction
import numpy as np

class ShootReward(BaseRewardFunction):
    """
    Reward for destroying an opponent by cannon or missile. 
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_kill = getattr(self.config, f'{self.__class__.__name__}_orientation_kill', True)

    def get_reward(self, task, env, agent_id, action=None):
        reward = 0

        # MISSILES NOT USED YET. 
        # for missile in env.agents[agent_id].missiles:
        #     if missile.success:
        #         reward += 5 #10
        #         # env.agents[agent_id].missiles.remove(missile)

        if env.agents[agent_id].cannon:
            if env.agents[agent_id].cannon.success:

                # encourage agent to kill opponent from behind. 
                if self.orientation_kill:
                    _, a_t, _, _, _, _ = super().get_orientation_metrics(env, agent_id, deg=False)
                    # scale, shift = task.shift_range(-1, 1, 5, 10)
                    scale, shift = task.shift_range(-1, 1, 0, 10)
                    orientation = np.cos(a_t)*scale + shift
                    reward += float(orientation)
                else:
                    reward += 5 #10
        
        return self._process(reward, agent_id)
