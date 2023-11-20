import rlcard
from rlcard.agents import RandomAgent, CFRAgent
from rlcard.models.pretrained_models import ROOT_PATH
from rlcard.utils import print_card, reorganize
import numpy as np
import os

def setupEnvironment(num_chips, custom_agent=None):
    # make environment
    global_seed = 0
    env = rlcard.make('no-limit-holdem', config={'seed': global_seed,'game_num_players': 2,'chips_for_each': num_chips})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': global_seed,'game_num_players': 2,'chips_for_each': num_chips})

    # set iteration numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000

    # set intial memory size
    memory_init_size = 1000
    max_buffer_size = 10000

    # train the agent every X steps
    train_every = 10

    # directory for logs
    log_dir = f'./experiments/logs/{evaluate_every}/'

    opponent_agent = CFRAgent(env, model_path=os.path.join(ROOT_PATH, 'leduc_holdem_cfr'))
    def step(state):
        action, _ = opponent_agent.eval_step(state)
        return action
    opponent_agent.step = step

    if custom_agent is not None:
        env.set_agents([custom_agent, opponent_agent])
    else:
        env.set_agents([RandomAgent(env.num_actions), opponent_agent])
    
    return env

def playGame(env, num_episodes, convergence_interval, is_training=True, training_data_filename='{}/../training_data/data_samples.csv'.format(os.path.dirname(__file__))):
    try:
        if is_training:
            os.makedirs(os.path.dirname(training_data_filename), exist_ok=True)
            f = open(training_data_filename, 'w')
            f.write('s,a,r,s_prime\n')
        
        for i in range(num_episodes):
            print("Game number " + str(i))
            print(">> Start a new game")

            env.agents[0].update_target_network()

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
            for j in range(1, len(action_record)+1):
                if action_record[-j][0] == state['current_player']:
                    break
                _action_list.insert(0, action_record[-j])
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

            # calculate convergence rate
            if i % convergence_interval == 0:
                convergence_rate = env.agents[0].calculate_convergence_rate()
                # env.agents[0].update_target_network()
                print(f'Convergence rate after {i} episodes: {convergence_rate}')

            if not is_training:
                input("Press any key to continue...")
    finally:
        if is_training:
            f.close()
