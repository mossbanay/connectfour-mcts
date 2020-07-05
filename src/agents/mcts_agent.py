from math import sqrt, log
from agents import Agent, RandomAgent

class MCTSNode():
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.n_visited = 0
        self.sum_value = 0

    @property
    def value(self):
        vhat = self.sum_value / self.n_visited
        explore = self.c * sqrt(log(self.parent.n_visited) / self.n_visited)
        return self.sum_value / self.n_visited

    @property
    def best_child(self):
        best_val = float('-inf')
        node = None

        for action, child in self.children:
            if child.value > best_val:
                node = child

        return node

    @property
    def best_action(self):
        best_val = float('-inf')
        best_action = None

        for action, child in self.children:
            if child.value > best_val:
                best_action = action

        return best_action

    def add_child(self, action, node):
        self.children.append((action, node))

class RandomAgent(Agent):
    def __init__(self, action_space, model):
        self.action_space = action_space
        self.model = model

        self.policy = {}

        # State
        self.prev_nodes = []

    def select_action(self, observation):
        if observation not in self.policy:
            # Create an empty node
            self.policy[observation] = Node(observation, None)

        cur_node = self.policy[observation]

        # Find leaf node by looking at the best for each level
        while len(cur_node.children) > 0:
            cur_node = cur_node.best_child

        if cur_node.n_visited == 0:
            # If the leaf node has not been visited before, do a rollout
            self.rollout(cur_node)
        else:
            # If it has been visited before, expand the children
            self.model.set_state(observation)
            for action in self.model.legal_moves():
                cur_node.add_child(action, Node(cur_node))

            # Pick an arbitrary child and do a rollout
            self.rollout(cur_node.best_child)

        if observation in self.policy:
            cur_node = self.policy[observation]
            if len(cur_node.children) > 0:
                return cur_node.best_action

        return self.action_space.sample()

    def observe_first(self, observation):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self):
        pass

    def rollout(self, state):
        pass
