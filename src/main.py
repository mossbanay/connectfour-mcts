import gym
import gym_connectfour


def main():
    env = gym.make('ConnectFour-v0')

    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 5, 3]
    move_idx = 0

    observation = env.reset()
    done = False

    while not done and move_idx < len(moves):
        move = moves[move_idx]
        move_idx += 1

        observation, reward, done, info = env.step(move)

        print(env.render())
        print(reward, done, info)
        print(f'Legal moves: {env.legal_moves()}')

if __name__ == '__main__':
    main()
