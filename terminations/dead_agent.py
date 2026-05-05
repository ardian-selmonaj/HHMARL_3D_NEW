from terminations.termination_base import BaseTerminationCondition


class DeadAgent(BaseTerminationCondition):
    """
    End up the simulation if the current aircraft has been shot down.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        if env.agents[agent_id].shotdown:
            self.log.info(f'{agent_id} has been shot down! Total Steps={env.steps}')
            return True, False, info

        elif env.agents[agent_id].crashed:
            self.log.info(f'{agent_id} has crashed! Total Steps={env.steps}')
            return True, False, info

        else:
            return False, False, info
