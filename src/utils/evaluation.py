class Evaluator:
    def __init__(self, env):
        self.env = env
        self.cached_metrics = None

    def run_evaluation_loop(self, num_episodes):
        total_wins = 0
        total_earnings = 0

        for _ in range(num_episodes):
            _, payoffs = self.env.run(is_training=False)
            total_earnings += payoffs[0]
            if payoffs[0] > 0:
                total_wins += 1

        win_rate = total_wins / num_episodes
        expected_earnings = total_earnings / num_episodes

        return win_rate, expected_earnings

    def calculate_win_rate(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[0]

    def calculate_expected_earnings(self, num_episodes):
        if self.cached_metrics is None:
            self.cached_metrics = self.run_evaluation_loop(num_episodes)
        return self.cached_metrics[1]