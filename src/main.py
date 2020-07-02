import gym
import gym_connectfour

import argparse
import pandas as pd

from tqdm import trange

from agents import Agent, RandomAgent


def play_game(env, first_agent: Agent, second_agent: Agent):
    """Pit two agents again each other, returning the final reward of the game"""

    observation = env.reset()
    timestep = (observation, 0, False, {})

    first_agent.observe_first(timestep)
    action = first_agent.select_action(observation)

    timestep = env.step(action)
    observation, reward, done, info = timestep

    first_agent.observe(action, timestep)
    second_agent.observe_first(timestep)

    action = second_agent.select_action(observation)
    timestep = env.step(action)
    observation, reward, done, info = timestep

    second_agent.observe(action, timestep)

    agent_idx = 0
    agents = [first_agent, second_agent]
    while not done:
        action = agents[agent_idx].select_action(observation)
        timestep = env.step(action)
        observation, reward, done, info = timestep
        agents[agent_idx].observe(action, timestep)
        agent_idx = (agent_idx + 1) % 2

    return reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-games', type=int, default=1000)

    args = parser.parse_args()

    env = gym.make('ConnectFour-v0')
    first_agent = RandomAgent(env.action_space)
    second_agent = RandomAgent(env.action_space)

    data = []
    for i in trange(args.n_games):
        reward = play_game(env, first_agent, second_agent)
        data.append({
            'game_idx': i,
            'reward': reward,
        })

    df = pd.DataFrame(data)
    print(df.describe())


if __name__ == '__main__':
    main()
