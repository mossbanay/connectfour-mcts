import gym
import gym_connectfour

import argparse
import pandas as pd

from tqdm import trange

from agents import Agent, RandomAgent


def play_game(env, agents):
    """Pit two agents again each other, returning the final reward of the game"""

    timestep = (env.reset(), 0, False, {})
    observation, reward, done, info = timestep
    last_action = []
    agent_idx = 0
    steps = 0

    for agent in agents:
        agent.observe_first(timestep)

        action = agent.select_action(observation)
        while action not in env.legal_moves():
            action = agent.select_action(observation)
        last_action.append(action)

        timestep = env.step(action)
        observation, reward, done, info = timestep

        steps += 1

    while not done:
        agent = agents[agent_idx]

        agent.observe(last_action[agent_idx], timestep)

        action = agent.select_action(observation)
        while action not in env.legal_moves():
            action = agent.select_action(observation)
        last_action[agent_idx] = action

        timestep = env.step(action)
        observation, reward, done, info = timestep

        agent_idx = (agent_idx + 1) % len(agents)
        steps += 1

    return {
        'reward': reward,
        'steps': steps,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-games', type=int, default=1000)

    args = parser.parse_args()

    env = gym.make('ConnectFour-v0')
    first_agent = RandomAgent(env.action_space)
    second_agent = RandomAgent(env.action_space)

    data = []
    for i in trange(args.n_games):
        result = play_game(env, [first_agent, second_agent])
        data.append({
            'game_idx': i,
            **result,
        })

    df = pd.DataFrame(data)
    print(df.describe())
    print(df['reward'].value_counts())


if __name__ == '__main__':
    main()
