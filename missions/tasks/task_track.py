from missions.mission_base import BaseMission
from rewards import ShootPenaltyReward, EventDrivenReward, HeadingReward, RollPitch
from terminations import HeadingReached

"""
Engage Task to track (pursue) target.
CannonTask and EscapeTask use this class to overwrite.

Cannon shots simulated through cone shaped Weapon Engagement Zone (WEZ).
"""

class TrackTask(BaseMission):
    def __init__(self, config):
        super().__init__(config)

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config), 
            RollPitch(self.config)
        ]

    # def load_terminations(self):
    #     super().load_terminations()
    #     self.terminations.append(HeadingReached(self.config))

    def get_action(self, action, agent, env):
        """
        Handle action processing.
        """
        shoot_action = action[4]
        action = super().get_action(action, agent, env)
        self._cannon(shoot_action, agent, env)
        return action
    
    def _cannon(self, shoot_action, agent, env):
        """
        Handle cannon action.
        """
        shoot_flag = agent.alive and shoot_action and agent.cannon.ready
        if shoot_flag: 
            #fire cannon
            self.shot_fired[agent.aid] = agent.cannon.launch(parent=agent, env=env)
        else:
            self.shot_fired[agent.aid] = False

    def reset(self, env):
        """
        Reset current task. 
        shot_fired stores TRUE if cannon / missile as activated, else is FALSE.
        """
        self.shot_fired = {aid: False for aid in env.agents.keys()}
        return super().reset(env)
    
    def step(self, env, agent):
        """
        Cannon reset if hit.
        """
        if agent.cannon.done: agent.cannon.revert_cannon()