from terminations.termination_base import BaseTerminationCondition
from env.catalog import Catalog as c


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state or exits game boundaries (map size). 
    """

    def __init__(self, config):
        super().__init__(config)
        self.map_size = getattr(self.config, 'map_size', 50) #km

    def get_termination(self, task, env, agent_id, info={}):
        done = bool(env.agents[agent_id].get_property_value(c.detect_extreme_state)) or bool(env.agents[agent_id].get_distance_to_origin_2d() > (self.map_size/2)*1000)
        if done:
            env.agents[agent_id].crash()
            self.log.info(f'{agent_id} is on an extreme state! Total Steps={env.steps}')
        success = False
        return done, success, info
