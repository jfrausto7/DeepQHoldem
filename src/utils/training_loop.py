# epsilon-greedy strategy for exploration
def calculate_epsilon(episode, epsilon_initial, epsilon_final, epsilon_anneal_duration):
    if episode < epsilon_anneal_duration:
        # linear annealing of epsilon
        epsilon = epsilon_initial - (episode / epsilon_anneal_duration) * (epsilon_initial - epsilon_final)
    else:
        epsilon = epsilon_final
    return epsilon

def train(agent, env, num_episodes, update_freq):
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # TODO: this is a simple implementation. figure out how to fit RLCard into this using playGame() or something
        while not done:
            epsilon = calculate_epsilon(episode)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, next_state, reward, done)
            state = next_state

        if episode % update_freq == 0:
            agent.update_target_network()


        