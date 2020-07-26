import dm_env
from dm_env import specs

import numpy as np


N_WIDTH = 7
N_HEIGHT = 6
N_STREAK_WIN = 4

_ACTIONS = (*range(N_WIDTH),)


class ConnectFourEnv(dm_env.Environment):
    def __init__(self):
        self._board = np.zeros((2, N_HEIGHT * N_WIDTH), dtype=np.bool)
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
                    mask = np.zeros((N_WIDTH, N_HEIGHT), dtype=np.bool)

                    try:
                        for i in range(N_STREAK_WIN):
                            mask[x + i * dx, y + i * dy] = True
                        winner_masks.append(mask.reshape(-1).copy())
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
            self._board[target_player][target_cell] = True
            self._col_heights[action] += 1

        self._player_one_turn = not self._player_one_turn

        # Check for termination.
        if self.is_terminal():
            reward = 1.0 if self._winner == 0 else -1.0 if self._winner == 1 else 0.0
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=0.0, observation=self._observation())

    def is_terminal(self):
        if any(
            [((self._board[0] & mask) == mask).all() for mask in self._winner_masks]
        ):
            self._winner = 0
            return True
        elif any(
            [((self._board[1] & mask) == mask).all() for mask in self._winner_masks]
        ):
            self._winner = 1
            return True
        elif self._col_heights.sum() == N_HEIGHT * N_WIDTH:
            self._winner = None
            return True

        return False

    def _observation(self):
        return self._board.copy()

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""

        self._board = np.zeros((2, N_HEIGHT * N_WIDTH), dtype=np.int8)
        self._col_heights = np.zeros(N_WIDTH, dtype=np.int8)
        self._player_one_turn = True
        self._winner = 0
        self._reset_next_step = False

        return dm_env.restart(self._observation())

    def legal_moves(self):
        """Find the current moves that are legal"""

        return self._col_heights < N_HEIGHT

    def set_state(self, board, player_one_turn):
        self._board = board
        self._col_heights = (
            self._board[0].reshape(N_WIDTH, N_HEIGHT)
            + self._board[1].reshape(N_WIDTH, N_HEIGHT)
        ).sum(axis=1)
        self._player_one_turn = player_one_turn
        self._reset_next_step = False

    def observation_spec(self):
        """Returns the observation spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=sum(self._board.shape), name="board"
        )

    def action_spec(self):
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=len(_ACTIONS), name="action")
