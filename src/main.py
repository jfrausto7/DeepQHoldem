import os
import argparse
import json
from agents.DeepQAgent import DeepQAgent
from environment.environment import setupEnvironment, playGame
from utils.evaluation import Evaluator

config = {
    "episodes": 100000,
    "state_size": 77,
    "num_actions": 23,
    "convergence_interval": 1000,
    "human_episodes": 5
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
        "--convergence_interval",
        default=config["convergence_interval"],
        type=int,
        help="Interval for calculating and printing the convergence rate during training.",
        dest="convergence_interval",
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

    parser.add_argument(
        "--players",
        default=2,
        type=int,
        help="Number of players in the game.",
        dest="players",
    )

    parser.add_argument(
        "--chips",
        default=1000,
        type=int,
        help="Number of starting chips for each agent/player.",
        dest="chips",
    )

    parser.add_argument(
        "--dropout",
        default=0.2,
        type=float,
        help="Dropout rate for the DeepQAgent ANN.",
        dest="dropout",
    )

    parser.add_argument(
        "--learning-rate",
        default=0.001,
        type=float,
        help="Learning rate for the DeepQAgent.",
        dest="learning_rate",
    )

    parser.add_argument(
        "--gamma",
        default=0.95,
        type=float,
        help="Discount factor gamma for the DeepQAgent.",
        dest="gamma",
    )

    parser.add_argument(
        "--epsilon",
        default=0.1,
        type=float,
        help="Epsilon value for the DeepQAgent epsilon-greedy action selection.",
        dest="epsilon",
    )

    # TODO: add any needed args for parsing

    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    # TODO: fill in main function

    # directory to store all the files/data for this experiment
    experiment_dir = f"./experiments/episodes{args.episodes}_players{args.players}_chips{args.chips}_dropout{args.dropout}_lr{args.learning_rate}_gamma{args.gamma}_epsilon{args.epsilon}/"
    os.makedirs(experiment_dir, exist_ok=True)

    experiment_params = {
        "episodes": args.episodes,
        "players": args.players,
        "chips": args.chips,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon": args.epsilon
    }
    with open(experiment_dir + 'experiment_params.json', 'w') as f:
        f.write(json.dumps(experiment_params, indent=4))

    # instantiate agent & envioronment
    agent = DeepQAgent(config["state_size"],
                       int(config["state_size"] / 3 + config["num_actions"]), # (1/3) of state space + action space
                       config["num_actions"],
                       learning_rate=args.learning_rate,
                       gamma=args.gamma,
                       epsilon=args.epsilon)
    env = setupEnvironment(args.players, args.chips, agent)

    # populate training_data_file with data from playing against an automated agent
    if args.generate:
        playGame(env, args.episodes, args.convergence_interval, experiment_dir, is_training=True)

    # training loop
    # train(agent, env, args.episodes, args.freq)

    # start game
    # playGame(env, num_episodes=args.episodes, is_training=False)

    # evaluate
    if args.human:
        env = setupEnvironment(2, args.chips, agent, is_human=True)
        args.episodes = args.human_episodes
    evaluator = Evaluator(env)

    f = open(experiment_dir + 'evaluation_results.txt', 'w')

    # calculate win rate after training or data generation
    win_rate = evaluator.calculate_win_rate(args.episodes)
    f.write(f"Win Rate: {win_rate * 100:.2f}%\n")
    print(f"Win Rate: {win_rate * 100:.2f}%")

    # calculate expected earnings
    avg_expected_earnings = evaluator.calculate_expected_earnings(args.episodes)
    f.write(f'Average Expected Earnings: {avg_expected_earnings}\n')
    print(f'Average Expected Earnings: {avg_expected_earnings}')

    # calculate action entropy
    entropy = evaluator.calculate_action_entropy(args.episodes)
    f.write(f'Action Entropy: {entropy}\n')
    print(f'Action Entropy: {entropy}')

    # calculate initial action distribution
    intial_action_distribution = evaluator.calculate_initial_action_distribution(args.episodes)
    f.write(f'Initial Action Distribution: FOLD = {intial_action_distribution[0]}, CALL = {intial_action_distribution[1]}, RAISE = {intial_action_distribution[2]}, ALL IN = {intial_action_distribution[3]}\n')
    print(f'Initial Action Distribution: FOLD = {intial_action_distribution[0]}, CALL = {intial_action_distribution[1]}, RAISE = {intial_action_distribution[2]}, ALL IN = {intial_action_distribution[3]}')
    
    # calculate average initial raise fraction of pot
    average_initial_raise_fraction = evaluator.calculate_initial_raise_fraction(args.episodes)
    f.write(f'Average Initial Raise Fraction of Pot: {average_initial_raise_fraction}\n')
    print(f'Average Initial Raise Fraction of Pot: {average_initial_raise_fraction}')

    # TODO plot convergence rates
    f.write(f'Convergence Rates: {agent.convergence_rates}\n')
    print(f'Convergence Rates: {agent.convergence_rates}')

    f.close()

    return None # TODO figure out what to return if anything

if __name__ == "__main__":
    args = parse_args()
    main(args)