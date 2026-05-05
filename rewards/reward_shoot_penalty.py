from rewards.reward_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching cannon, give -0.1 penalty reward to be cautious with firing.
    """
    def __init__(self, config):
        super().__init__(config)
        self.posture_punish = getattr(self.config, f'{self.__class__.__name__}_posture_punish', False)

    def get_reward(self, task, env, aid, action=None):
        a_a, _, _, _, dist = super().get_orientation_metrics(env, aid, deg=True)
        reward = 0

        if self.posture_punish:
            # if in good shooting position but cannon fire is not active
            if a_a <= 10 and dist < 3600:
                # if not task.shot_fired[aid]:
                if not env.agents[aid].cannon.alive:
                    reward -= 0.1
            # if not in good shooting position but cannon fire is executed
            elif a_a > 90 or dist > 15000:
                if task.shot_fired[aid]:
                    reward -= 0.1
        else:
            # constant punishment for every executed cannon shot
            if task.shot_fired[aid]:
                reward -= 0.1

        return self._process(reward, aid)
        
