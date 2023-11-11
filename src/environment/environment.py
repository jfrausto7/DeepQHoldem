import rlcard
from rlcard.agents import RandomAgent, NolimitholdemHumanAgent as HumanAgent
from rlcard.utils import print_card, reorganize
import numpy as np
import os

def setupEnvironment(custom_agent=None):
    # make environment
    global_seed = 0
    env = rlcard.make('no-limit-holdem', config={'seed': global_seed,'game_num_players': 2,'chips_for_each': 1000})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': global_seed,'game_num_players': 2,'chips_for_each': 1000})

    # set iteration numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 100000

    # set intial memory size
    memory_init_size = 1000
    max_buffer_size = 10000

    # train the agent every X steps
    train_every = 10

    # directory for logs
    log_dir = f'./experiments/logs/{evaluate_every}/'

    if custom_agent is not None:
        env.set_agents([custom_agent, RandomAgent(env.num_actions)])
    else:
        env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    
    return env

def playGame(env, is_training=True):
    try:
        if is_training:
            filename = '{}/../training_data/data_samples.csv'.format(os.path.dirname(__file__))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            f = open(filename, 'w')
            f.write('s,a,r,s_prime\n')
        
        while (True):
            print(">> Start a new game")

            trajectories, payoffs = env.run(is_training=True)

            if is_training:
                reorganized_trajectories = reorganize(trajectories, payoffs)
                # assume the custom agent is player 1 (index 0)
                for traj in reorganized_trajectories[0]:
                    state, a, r, next_state, _ = traj
                    legal_actions = np.zeros(env.num_actions)
                    legal_actions[list(state['legal_actions'].keys())] = 1
                    s = np.concatenate((state['obs'], legal_actions))
                    next_legal_actions = np.zeros(env.num_actions)
                    next_legal_actions[list(next_state['legal_actions'].keys())] = 1
                    s_prime = np.concatenate((next_state['obs'], next_legal_actions))
                    f.write('{},{},{},{}\n'.format(s, a, r, s_prime))

            # If the human does not take the final action, we need to
            # print other players action
            final_state = trajectories[0][-1]
            action_record = final_state['action_record']
            state = final_state['raw_obs']
            _action_list = []
            for i in range(1, len(action_record)+1):
                if action_record[-i][0] == state['current_player']:
                    break
                _action_list.insert(0, action_record[-i])
            for pair in _action_list:
                print('>> Player', pair[0], 'chooses', pair[1])

            # Let's take a look at what the agent card is
            print('===============     Cards all Players 1   ===============')
            for hands in env.get_perfect_information()['hand_cards']:
                print_card(hands)

            print('===============     Result     ===============')
            if payoffs[0] > 0:
                print('You win {} chips!'.format(payoffs[0]))
            elif payoffs[0] == 0:
                print('It is a tie.')
            else:
                print('You lose {} chips!'.format(-payoffs[0]))
            print('')

            if not is_training:
                input("Press any key to continue...")
    finally:
        if is_training:
            f.close()
