from environment.environment import playGame

def generateData(env, num_episodes, convergence_interval, data_filename):
    playGame(env, num_episodes, convergence_interval, is_training=True, training_data_filename=data_filename)