import gym_connectfour

import argparse
import pandas as pd

from tqdm import trange

from agents import Agent, RandomAgent, MCTSAgent


def play_game(env, agent):
    """Play through one episode of the game, returning the final reward of the game"""

    timestep = env.reset()

    while True:
        action = agent.step(timestep)
        timestep = env.step(action)

        if timestep.last():
            _ = agent.step(timestep)
            break

    return {
        "reward": timestep.reward,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=1000)

    args = parser.parse_args()

    env = gym_connectfour.envs.ConnectFourEnv()

    # first_agent = MCTSAgent(env.action_spec())
    second_agent = RandomAgent(env.action_spec())

    data = []
    for i in trange(args.n_games):
        result = play_game(env, second_agent)  # [first_agent, second_agent])

        data.append(
            {"game_idx": i, **result,}
        )

    df = pd.DataFrame(data)
    print(df.describe())
    print(df["reward"].value_counts())


if __name__ == "__main__":
    main()
