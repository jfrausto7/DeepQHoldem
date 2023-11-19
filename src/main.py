import os
import argparse
import torch
from agents.DeepQAgent import DeepQAgent
from environment.environment import playGame, setupEnvironment

from utils.data_utils import generateData
from utils.evaluation import calculate_win_rate
from utils.training_loop import train


config = {
    "episodes": 100000,
    "chips": 1000,
    "state_size": 77,
    "num_actions": 23,
    "training_data_filename": '{}/training_data/data_samples.csv'.format(os.path.dirname(__file__))
}

def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our DeepQPoker project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--generate", help="Generate data using naive agent.", action="store_true", dest="generate"
    )

    parser.add_argument(
        "--episodes",
        default=config["episodes"],
        type=int,
        help="Number of episodes for training.",
        dest="episodes",
    )

    parser.add_argument(
        "--chips",
        default=config["episodes"],
        type=int,
        help="Number of starting chips for each agent/player.",
        dest="chips",
    )

    # TODO: add any needed args for parsing

    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    # TODO: fill in main function

    # instantiate agent & envioronment
    agent = DeepQAgent(config["state_size"], int(config["state_size"] / 3 + config["num_actions"]), config["num_actions"]) # (1/3) of state space + action space
    env = setupEnvironment(num_chips=args.chips, custom_agent=agent)

    # populate training_data_file with data from playing against an automated agent
    if args.generate:
        generateData(env, args.episodes, config["training_data_filename"])

    # training loop
    # train(agent, env, args.episodes, args.freq)

    # start game
    # playGame(env, num_episodes=args.episodes, is_training=False)

    # calculate win rate after training or data generation
    win_rate = calculate_win_rate(env, args.episodes)
    print(f"Win Rate: {win_rate * 100:.2f}%")

    return None # TODO figure out what to return if anything

if __name__ == "__main__":
    args = parse_args()
    main(args)