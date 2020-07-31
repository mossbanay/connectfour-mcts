from random import choice
from gym_connectfour import ConnectFourEnv

import time

from math import sqrt, log
from agents import RandomAgent

from collections import namedtuple


EPSILON = 1e-5
ValueInfo = namedtuple("ValueInfo", "n_visited sum_value")


class MCTSNode:
    def __init__(self, observation, c=sqrt(2)):
        self.n_visited = 0
        self.c = c

        self.legal_moves = ConnectFourEnv.get_legal_moves(observation)
        self.children = {action: ValueInfo(0, 0) for action in self.legal_moves}

        if self.legal_moves == []:
            print("uh oh")

    def best_action(self):
        best_value = float("-inf")
        best_act = None

        for action, value_info in self.children.items():
            v = value_info.sum_value / (value_info.n_visited + EPSILON)
            v += (
                self.c
                * sqrt(log(self.n_visited + EPSILON) / (value_info.n_visited + EPSILON))
                if self.n_visited > 0
                else 0
            )

            if v > best_value:
                best_value = v
                best_act = action

        if best_act is None:
            print("uh oh")

        return best_act

    def get_leaf_actions(self):
        leaf_actions = []

        for action, value_info in self.children.items():
            if value_info.n_visited == 0:
                leaf_actions.append(action)

        return leaf_actions

    def update(self, action, win):
        value_info = self.children[action]
        self.children[action] = ValueInfo(
            value_info.n_visited + 1, value_info.sum_value + win
        )


class MCTSAgent:
    def __init__(self, action_spec, time_budget):
        self.action_spec = action_spec

        self.sim_env = ConnectFourEnv()
        self.opponent_agent = RandomAgent(action_spec)

        self.time_budget = time_budget

        self.policy = {}

    def step(self, timestep):
        if timestep.last():
            return 0

        # Ensure the current state is added into policy
        if timestep.observation not in self.policy:
            self.policy[timestep.observation] = MCTSNode(timestep.observation)

        self.policy[timestep.observation].best_action()

        begin_time = time.time()

        # Run rollouts
        n_updates = 0
        while time.time() < begin_time + self.time_budget:
            self.update(timestep.observation)
            n_updates += 1

        # print(f"Ran {n_updates} in {time.time() - begin_time:.2f}s")

        return self.policy[timestep.observation].best_action()

    def update(self, observation):
        # Set simulator to the same state
        self.sim_env.set_state(observation, True)

        # The path we take is a list of (observation, action) tuples
        path = []
        timestep = None

        self.policy[observation].best_action()

        # We should be guaranteed to be able to observe at least one more timestep
        # since we should not have been passed in a final timestep
        while True:
            if observation not in self.policy:
                # Expand this observation
                node = MCTSNode(observation)
                if node.get_leaf_actions() == []:
                    break
                self.policy[observation] = node
                action = node.get_leaf_actions()[0]
                path.append((observation, action))
                timestep = self.rollout(self.sim_env.step(action))
                break

            cur_node = self.policy[observation]
            leaf_actions = cur_node.get_leaf_actions()

            action = None

            if len(leaf_actions) == 0:
                # If we've expanded all of this node, select the best action and take a step in env
                action = cur_node.best_action()
            else:
                # Expand the first leaf in this node
                action = leaf_actions[0]

            path.append((observation, action))
            timestep = self.sim_env.step(action)
            if timestep.last():
                break
            timestep = self.sim_env.step(self.opponent_agent.step(timestep))
            if timestep.last():
                break

            observation = timestep.observation

        # Backpropagation
        for obs, action in path:
            self.policy[obs].update(action, timestep.reward)

    def rollout(self, timestep):
        while not timestep.last():
            timestep = self.sim_env.step(
                choice(ConnectFourEnv.get_legal_moves(timestep.observation))
            )

        return timestep

    def set_time_budget(self, time_budget):
        self.time_budget = time_budget
