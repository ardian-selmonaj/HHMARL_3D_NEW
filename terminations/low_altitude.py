from terminations.termination_base import BaseTerminationCondition
from env.catalog import Catalog as c


class LowAltitude(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low or too high.
    """

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m
        self.altitude_limit_max = getattr(config, 'altitude_limit_max', 10000)  # unit: m

    def get_termination(self, task, env, agent_id, info={}):
        height = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        done = height <= self.altitude_limit or height > self.altitude_limit_max
        if done:
            env.agents[agent_id].crash()
            self.log.info(f'{agent_id} altitude is too low. Total Steps={env.steps}')
        success = False
        return done, success, info
