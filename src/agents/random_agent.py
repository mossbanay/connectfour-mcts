from agents import Agent


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()

    def observe_first(self, observation):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self):
        pass