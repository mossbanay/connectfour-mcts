import abc


class Agent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, observation):
        """Sample from policy and return an action"""

    @abc.abstractmethod
    def observe_first(self, timestep):
        """Process initial state of environment"""

    @abc.abstractmethod
    def observe(self, action, next_timestep):
        """Process action and subsequent timestep"""
    
    @abc.abstractmethod
    def update(self):
        """Perform an update on policy"""
