import os
import argparse
import torch
from agents.DeepQAgent import DeepQAgent
from environment.environment import playGame, setupEnvironment

from utils.data_utils import generateData
from utils.training_loop import train


config = {
    "episodes": 500,

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
        "--freq",
        default=config["freq"],
        type=int,
        help="Number of episodes to update network with.",
        dest="freq",
    )

    # TODO: add any needed args for parsing

    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    # TODO: fill in main function

    if args.generate:
        data = generateData()

    # instantiate agent & envioronment
    agent = DeepQAgent()
    env = setupEnvironment()

    # training loop
    train(agent, env, args.episodes, args.freq)

    # start game
    # playGame()

    return None # TODO figure out what to return if anything

if __name__ == "__main__":
    args = parse_args()
    main(args)