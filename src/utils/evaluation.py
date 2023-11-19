def calculate_win_rate(env, num_episodes):
    wins = 0
    for _ in range(num_episodes):
        _, payoffs = env.run(is_training=False)
        if payoffs[0] > 0:
            wins += 1
    win_rate = wins / num_episodes
    return win_rate
