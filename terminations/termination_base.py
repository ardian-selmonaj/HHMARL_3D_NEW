import logging
from abc import ABC, abstractmethod


class BaseTerminationCondition(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.config = config

    @abstractmethod
    def get_termination(self, task, env, agent_id, info={}):
        """
        Determine whether the episode should terminate.
        """
        raise NotImplementedError
