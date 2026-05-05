from missions import TrackTask
from rewards import ShootPenaltyReward, ShootReward, EventDrivenReward, HeadingReward, RollPitch

"""
Attack Task for aggressively destroying opponents.
"""

class CannonTask(TrackTask):
    def __init__(self, config):
        super().__init__(config)

    def load_rewards(self):
        self.rewards = [
            ShootReward(self.config),
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            HeadingReward(self.config),
            #RollPitch(self.config)
        ]