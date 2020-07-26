import random

from agents import Agent


class RandomAgent:
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def step(self, timestep):
        return random.randrange(self.action_spec.minimum, self.action_spec.maximum + 1)

    def update(self):
        pass
