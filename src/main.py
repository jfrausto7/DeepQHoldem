import os
import argparse
from agents.LAGAgent import LAGAgent
from agents.TAGAgent import TAGAgent
import torch
from agents.DeepQAgent import DeepQAgent
from environment.environment import setupEnvironment

from utils.data_utils import generateData
from utils.evaluation import Evaluator

config = {
    "episodes": 100000,
    "chips": 1000,
    "state_size": 77,
    "num_actions": 23,
    "convergence_interval": 1000,
    "human_episodes": 5,
    "adversarial_agent": None,
    "training_data_filename": '{}/training_data/data_samples.csv'.format(os.path.dirname(__file__))
}

def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our DeepQHoldem project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--generate", help="Generate data & train with agent.", action="store_true", dest="generate"
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
        default=config["chips"],
        type=int,
        help="Number of starting chips for each agent/player.",
        dest="chips",
    )

    parser.add_argument(
        "--convergence_interval",
        default=config["convergence_interval"],
        type=int,
        help="Interval for calculating and printing the convergence rate during training.",
        dest="convergence_interval",
    )

    parser.add_argument(
        "--adversary",
        default=config["adversarial_agent"],
        type=int,
        help="Integer representing adversarial agent to evaluate against.",
        dest="adversary",
    )

    parser.add_argument(
        "--human", help="Play against the agent! Be sure to include the '--generated' flag otherwise it won't be trained!",
        action="store_true", dest="human"
    )

    parser.add_argument(
        "--human_episodes",
        default=config["human_episodes"],
        type=int,
        help="Number of rounds to play against agent when using '--human' flag.",
        dest="human_episodes",
    )

    # TODO: add any needed args for parsing

    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    # TODO: fill in main function

    # instantiate agent & envioronment
    agent = DeepQAgent(config["state_size"], int(config["state_size"] / 3 + config["num_actions"]), config["num_actions"]) # (1/3) of state space + action space
    
    envs = []

    # error checking
    if args.adversary is not None and args.adversary > 5:
        raise Exception("Adversary is out of range! Enter a number between 1-5.")
    
    # adversarial agent strategies
    if args.adversary == 1:
        adversarial_agent = TAGAgent(config["state_size"], int(config["state_size"] / 3 + config["num_actions"]), config["num_actions"]) # (1/3) of state space + action space
        env = setupEnvironment(num_chips=args.chips, custom_agent=agent, custom_adversary=adversarial_agent)
        envs.append((env, "TAG"))
    elif args.adversary == 2:
        adversarial_agent = LAGAgent(config["state_size"], int(config["state_size"] / 3 + config["num_actions"]), config["num_actions"]) # (1/3) of state space + action space
        env = setupEnvironment(num_chips=args.chips, custom_agent=agent, custom_adversary=adversarial_agent)
        envs.append((env, "LAG"))
    else:
        env = setupEnvironment(num_chips=args.chips, custom_agent=agent)

    # play and learn against agent
    if args.generate:
        generateData(env, args.episodes, args.convergence_interval, config["training_data_filename"])

    # evaluate
    if args.human:
        env = setupEnvironment(num_chips=args.chips, custom_agent=agent, is_human=True)
        args.episodes = args.human_episodes
        envs.append((env, "Human"))
    else:
        randomEnv = setupEnvironment(num_chips=args.chips, custom_agent=agent)
        envs.append((randomEnv, "Random"))
        leducEnv = setupEnvironment(num_chips=args.chips, custom_agent=agent, is_leduc=True)
        envs.append((leducEnv, "Leduc"))

    # calculate win rate after training or data generation
    for env in envs:
        evaluator = Evaluator(env[0])
        win_rate = evaluator.calculate_win_rate(args.episodes)
        print(f"Win Rate against {env[1]} agent: {win_rate * 100:.2f}%")

        # calculate expected earnings
        avg_expected_earnings = evaluator.calculate_expected_earnings(args.episodes)
        print(f'Average Expected Earnings against {env[1]} agent: {avg_expected_earnings}')

        # calculate action entropy
        entropy = evaluator.calculate_action_entropy(args.episodes)
        print(f'Action Entropy against {env[1]} agent: {entropy}')

        # calculate initial action distribution
        intial_action_distribution = evaluator.calculate_initial_action_distribution(args.episodes)
        print(f'Initial Action Distribution against {env[1]} agent: FOLD = {intial_action_distribution[0]}, CALL = {intial_action_distribution[1]}, RAISE = {intial_action_distribution[2]}, ALL IN = {intial_action_distribution[3]}')
        
        # calculate average initial raise fraction of pot
        average_initial_raise_fraction = evaluator.calculate_initial_raise_fraction(args.episodes)
        print(f'Average Initial Raise Fraction of Pot against {env[1]} agent: {average_initial_raise_fraction}')

        # TODO plot convergence rates
        print(f'Convergence Rates against {env[1]} agent: {agent.convergence_rates}')

    return None # TODO figure out what to return if anything

if __name__ == "__main__":
    args = parse_args()
    main(args)