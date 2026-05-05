from missions import TrackTask
from rewards import ShootPenaltyReward, EventDrivenReward, DistanceReward

"""
Defend Task to escape from opponent.
"""

class EscapeTask(TrackTask):
    def __init__(self, config):
        super().__init__(config)

    def load_rewards(self):
        self.rewards = [
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config),
            DistanceReward(self.config)
        ]