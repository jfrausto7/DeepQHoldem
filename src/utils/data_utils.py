from environment.environment import playGame

def generateData(env, num_episodes, data_filename):
    playGame(env, num_episodes, is_training=True, training_data_filename=data_filename)