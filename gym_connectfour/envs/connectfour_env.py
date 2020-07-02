import gym
from gym import spaces


N_WIDTH = 7
N_HEIGHT = 6
N_STREAK_WIN = 4


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}
    action_space = spaces.Discrete(N_WIDTH)
    observation_space = spaces.Tuple((spaces.MultiDiscrete([N_HEIGHT, N_WIDTH, 3]), spaces.Discrete(2)))
    reward_range = (-1, 1)

    def __init__(self):
        self.player_one_turn = True
        self.board = []
        self.reset()

    def step(self, action):
        """Play out one action in the environment"""
        
        for i in range(N_HEIGHT):
            if self.board[i][action] == 0:
                self.board[i][action] = 1 if self.player_one_turn else 2
                break

        self.player_one_turn = not self.player_one_turn

        observation = (self.board, self.player_one_turn)
        winner = self.winner()
        done = winner is not None or self.legal_moves() == []
        reward = 1 if winner == 1 else -1 if winner == 2 else 0
        info = {}

        return (observation, reward, done, info)

    def reset(self):
        """Reset the environment"""

        self.player_one_turn = True
        self.board = [[0 for _ in range(N_WIDTH)] for _ in range(N_HEIGHT)]

    def legal_moves(self):
        """Find the current moves that are legal"""

        moves = []
        for i in range(N_WIDTH):
            if self.board[N_HEIGHT-1][i] == 0:
                moves.append(i)
        
        return moves

    def winner(self):
        """Determine if one of the players has won the game. Returning 1 or 2 if so and None otherwise"""
        # Check for a horizontal win
        for i in range(N_HEIGHT):
            streak = 0
            last = 0
            for j in range(N_WIDTH):
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 0
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
        
        # Check for a vertical win
        for j in range(N_WIDTH):
            streak = 0
            last = 0
            for i in range(N_HEIGHT):
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 1
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
        
        # Check upward sloping diagonals
        for i in range(N_HEIGHT-1, 0, -1):
            j = 0
            streak = 0
            last = 0
            while i < N_HEIGHT and j < N_WIDTH:
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 1
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
                i += 1
                j += 1
        
        for j in range(N_WIDTH):
            i = 0
            streak = 0
            last = 0
            while i < N_HEIGHT and j < N_WIDTH:
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 1
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
                i += 1
                j += 1
  
        # Check downward sloping diagonals
        for i in range(0, N_HEIGHT-1):
            j = N_WIDTH-1
            streak = 0
            last = 0
            while i > 0 and j > 0:
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 1
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
                i -= 1
                j -= 1
        
        for j in range(N_WIDTH):
            i = N_HEIGHT-1
            streak = 0
            last = 0
            while i > 0 and j > 0:
                current = self.board[i][j]
                if current == last:
                    streak += 1
                else:
                    streak = 1
                
                if current != 0 and streak == N_STREAK_WIN:
                    return last

                last = current
                i -= 1
                j -= 1
        
        return None

    def render(self, mode='ansi', close=False):
        """Render the board as a string"""

        if mode != 'ansi':
            raise NotImplementedError("Only ansi render mode is supported.")

        num_to_char = {
            0: ' ',
            1: '1',
            2: '2',
        }

        rows = []

        next_to_move = 'Player 1' if self.player_one_turn else 'Player 2'
        rows.append(f'{next_to_move} to move')

        rows.append('╚' + '═══╩'*(N_WIDTH-1) + '═══╝')
        rows.extend(['║ ' + ' ║ '.join(map(lambda x: num_to_char[x], row)) + ' ║' for row in self.board])


        return '\n'.join(rows[::-1])
