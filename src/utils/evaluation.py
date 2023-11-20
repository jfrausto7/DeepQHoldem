import numpy as np

class Evaluator:
    def __init__(self, env):
        self.env = env
        self.cached_metrics = None

    def run_evaluation_loop(self, num_episodes):
        total_wins = 0
        total_earnings = 0
        action_probabilities = []

        for _ in range(num_episodes):
            _, payoffs = self.env.run(is_training=False)
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

        return win_rate, expected_earnings, entropy

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
