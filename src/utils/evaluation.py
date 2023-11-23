import numpy as np
from rlcard.utils import reorganize
from rlcard.games.nolimitholdem.round import Action

class Evaluator:
    def __init__(self, env):
        self.env = env
        self.cached_metrics = None

    def run_evaluation_loop(self, num_episodes):
        total_wins = 0
        total_earnings = 0
        action_probabilities = []
        initial_actions = 0
        initial_folds = 0
        initial_calls = 0
        initial_raises = 0
        initial_all_ins = 0
        average_initial_raise_fraction = 0

        for _ in range(num_episodes):
            trajectories, payoffs = self.env.run(is_training=False)

            reorganized_trajectories = reorganize(trajectories, payoffs)
            action_record = list(filter(lambda action : action[0] == 0, reorganized_trajectories[-1][-1][0]['action_record']))
            if len(action_record) > 0:
                initial_actions += 1
                if action_record[0][1] == Action.FOLD:
                    initial_folds += 1
                elif action_record[0][1] == Action.CHECK_CALL:
                    initial_calls += 1
                elif action_record[0][1] == Action.ALL_IN:
                    initial_all_ins += 1
                else:
                    initial_raises += 1
                    pct = int(action_record[0][1].name[action_record[0][1].name.index("_") + 1 : action_record[0][1].name.index("PCT")])
                    average_initial_raise_fraction = (((initial_raises - 1) * average_initial_raise_fraction) + pct) / initial_raises
            
            total_earnings += payoffs[0]
            if payoffs[0] > 0:
                total_wins += 1

            num_actions = len(self.env.get_perfect_information()['legal_actions'])
            uniform_prob = 1.0 / num_actions  # assuming uniform distribution over actions
            action_probabilities.append(uniform_prob)
            # action_probabilities.extend(self.env.get_perfect_information()['probs'])

        win_rate = total_wins / num_episodes
        expected_earnings = total_earnings / num_episodes
        entropy = -np.sum(action_probabilities * np.log(action_probabilities)) / num_episodes
        initial_action_distribution = np.array([initial_folds, initial_calls, initial_raises, initial_all_ins]) / initial_actions

        return win_rate, expected_earnings, entropy, initial_action_distribution, average_initial_raise_fraction

    def calculate_win_rate(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[0]

    def calculate_expected_earnings(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[1]

    def calculate_action_entropy(self, num_episodes):
        """Range of values are 0 to log_2(N). Lower values = Deterministic;
        Higher values = Stochastic."""
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[2]

    def calculate_initial_action_distribution(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[3]
    
    def calculate_initial_raise_fraction(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[4]
