import dm_env
from dm_env import specs

import numpy as np


N_WIDTH = 7
N_HEIGHT = 6
N_STREAK_WIN = 4

_ACTIONS = (*range(N_WIDTH),)


def popcount(x):
    b = 0
    while x > 0:
        x &= x - 1
        b += 1
    return b


class ConnectFourEnv(dm_env.Environment):
    def __init__(self):
        self._board = [0, 0]
        self._col_heights = np.zeros(N_WIDTH, dtype=np.int8)
        self._player_one_turn = True
        self._winner = None
        self._reset_next_step = True

        # Precompute masks of winning positions
        self._winner_masks = self.generate_winner_masks()

    def generate_winner_masks(self):
        winner_masks = []
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for (dx, dy) in dirs:
            for x in range(N_WIDTH):
                for y in range(N_HEIGHT):
                    mask = 0

                    try:
                        for i in range(N_STREAK_WIN):
                            mask |= 1 << ((x + i * dx) * N_HEIGHT + y + i * dy)

                        winner_masks.append(mask)
                    except IndexError:
                        pass

        return winner_masks

    def step(self, action):
        """Updates the environment according to the action."""

        if self._reset_next_step:
            return self.reset()

        # Insert token if column isn't full if column is full
        if self._col_heights[action] < N_HEIGHT:
            target_cell = action * N_HEIGHT + self._col_heights[action]
            target_player = 0 if self._player_one_turn else 1
            self._board[target_player] |= 1 << target_cell
            self._col_heights[action] += 1
        else:
            print("Illegal move!")

        self._player_one_turn = not self._player_one_turn

        # Check for termination.
        if self.is_terminal():
            reward = 1.0 if self._winner == 0 else -1.0 if self._winner == 1 else 0.0
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=0.0, observation=self._observation())

    def is_terminal(self):
        if any((mask & self._board[0]) == mask for mask in self._winner_masks):
            self._winner = 0
            return True
        elif any((mask & self._board[1]) == mask for mask in self._winner_masks):
            self._winner = 1
            return True
        elif self._col_heights.sum() == N_HEIGHT * N_WIDTH:
            self._winner = None
            return True

        return False

    def _observation(self):
        return tuple(self._board)

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""

        self._board = [0, 0]
        self._col_heights = np.zeros(N_WIDTH, dtype=np.int8)
        self._player_one_turn = True
        self._winner = 0
        self._reset_next_step = False

        return dm_env.restart(self._observation())

    def set_state(self, board, player_one_turn):
        self._board = list(board)

        col_mask = (1 << N_HEIGHT) - 1
        self._col_heights = []

        full_board = board[0] + board[1]

        for _ in range(N_WIDTH):
            self._col_heights.append(popcount(col_mask & full_board))
            col_mask <<= N_HEIGHT

        self._col_heights = np.array(self._col_heights)
        self._player_one_turn = player_one_turn
        self._reset_next_step = False

    def observation_spec(self):
        """Returns the observation spec."""
        return specs.DiscreteArray(dtype=int, num_values=2, name="board")

    def action_spec(self):
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=len(_ACTIONS), name="action")

    @staticmethod
    def get_legal_moves(observation):
        board = observation[0] + observation[1]
        top_slot = 1 << (N_HEIGHT - 1)
        moves = []

        for idx in range(N_WIDTH):
            if (board & top_slot) != top_slot:
                moves.append(idx)

            top_slot <<= N_HEIGHT

        return moves
