from random import choice
from gym_connectfour import ConnectFourEnv

from agents import Agent


class RandomAgent:
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def step(self, timestep):
        if timestep.last():
            return 0

        legal_moves = ConnectFourEnv.get_legal_moves(timestep.observation)
        return choice(legal_moves)

    def update(self):
        pass
