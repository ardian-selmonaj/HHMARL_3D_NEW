from terminations.termination_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)
        self.horizon = getattr(self.config, 'horizon', 1000)

    def get_termination(self, task, env, agent_id, info={}):
        done = env.steps >= self.horizon
        success = False
        return done, success, info
