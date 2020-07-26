import gym_connectfour

import argparse
import pandas as pd

from tqdm import trange

from agents import Agent, RandomAgent, MCTSAgent


def play_game(env, agent, opponent_agent=None):
    """Play through one episode of the game, returning the final reward of the game"""

    timestep = env.reset()

    while True:
        action = agent.step(timestep)
        timestep = env.step(action)

        if timestep.last():
            _ = agent.step(timestep)
            break

        if opponent_agent is not None:
            action = opponent_agent.step(timestep)
            timestep = env.step(action)
            agent.step(timestep)

            if timestep.last():
                break

    return {
        "reward": timestep.reward,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=1000)

    args = parser.parse_args()

    env = gym_connectfour.envs.ConnectFourEnv()

    agent = MCTSAgent(env.action_spec(), time_budget=1)
    random_agent = RandomAgent(env.action_spec())

    data = []
    for i in trange(args.n_games):
        result = play_game(env, agent, opponent_agent=random_agent)

        data.append(
            {"game_idx": i, **result,}
        )

    df = pd.DataFrame(data)
    print(df.describe())
    print(df["reward"].value_counts())


if __name__ == "__main__":
    main()
