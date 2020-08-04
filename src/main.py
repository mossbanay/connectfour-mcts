import gym_connectfour

import argparse
import pandas as pd
import seaborn as sns

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
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


def task(round_n, n_games=25):
    env = gym_connectfour.envs.ConnectFourEnv()
    random_agent = RandomAgent(env.action_spec())

    agent = MCTSAgent(env.action_spec(), time_budget=0.001)

    data = []

    for game_n in range(n_games):
        results = play_game(env, agent, opponent_agent=random_agent)

        data.append({"round_n": round_n, "game_n": game_n, **results})

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=25)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--n-threads", type=int, default=1)

    args = parser.parse_args()

    pool = Pool(args.n_threads)
    data = []
    t = partial(task, n_games=args.n_games)
    for step_data in tqdm(
        pool.imap_unordered(t, range(args.n_rounds)), total=args.n_rounds
    ):
        data.extend(step_data)

    df = pd.DataFrame(data)

    g = sns.relplot(
        x="game_n", y="reward", kind="line", height=4, aspect=10 / 4, data=df
    )
    g.set(xlabel="Games played", ylabel="Win rate", title="MCTS agent vs Random agent")
    g.savefig("output.png")


if __name__ == "__main__":
    main()
