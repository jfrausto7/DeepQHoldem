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

    # TODO: add any needed args for parsing

    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    # TODO: fill in main function

    if args.generate:
        data = generateData()

    # instantiate agent & envioronment
    agent = DeepQAgent(60, 60, 6) # TODO decide on middle value (first and last value should be defined by the state size and action space size respectively)
    env = setupEnvironment(custom_agent=agent)

    # training loop
    # train(agent, env, args.episodes, args.freq)

    # start game
    playGame(env, is_training=True)

    return None # TODO figure out what to return if anything

if __name__ == "__main__":
    args = parse_args()
    main(args)