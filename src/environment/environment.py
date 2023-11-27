from logging import Logger
import torch
import rlcard
from rlcard.agents import RandomAgent, CFRAgent
from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from rlcard.models.pretrained_models import ROOT_PATH
from rlcard.utils import print_card, reorganize, tournament, Logger, plot_curve
import numpy as np
import os

def setupEnvironment(num_chips, custom_agent=None, custom_adversary=None, is_human=False, is_leduc=False):
    # make environment
    global_seed = 0
    env = rlcard.make('no-limit-holdem', config={'seed': global_seed,'game_num_players': 2,'chips_for_each': num_chips})
    env.customAdversary = custom_adversary

    def step(state):
        action, _ = opponent_agent.eval_step(state)
        return action
    
    if not custom_adversary:
        if is_leduc:
            opponent_agent = CFRAgent(env, model_path=os.path.join(ROOT_PATH, 'leduc_holdem_cfr'))
            opponent_agent.step = step
        else:
            opponent_agent = RandomAgent(env.num_actions)
    else:
        opponent_agent = custom_adversary

    if is_human:
        opponent_agent = HumanAgent(env.num_actions)

    if custom_agent is not None:
        env.set_agents([custom_agent, opponent_agent])
    else:
        env.set_agents([RandomAgent(env.num_actions), opponent_agent])
    
    return env

def playGame(env, num_episodes, convergence_interval, is_training=True, training_data_filename='{}/../training_data/data_samples.csv'):

    log_dir = f'./experiments/logs/{num_episodes}/'
    try:
        if is_training:
            os.makedirs(os.path.dirname(training_data_filename), exist_ok=True)
            f = open(training_data_filename, 'w')
            f.write('s,a,r,s_prime\n')
        
        with Logger(log_dir) as logger:
            for i in range(num_episodes):
                print("Game number " + str(i))
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

                        # Train the agent online
                        env.agents[0].train(torch.from_numpy(s).float(), a, torch.from_numpy(s_prime).float(), r)

                # If the human does not take the final action, we need to
                # print other players' actions
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
                print('===============     Cards     ===============')
                for n, hands in enumerate(env.get_perfect_information()['hand_cards']):
                    print(f'Player {n+1}\'s Hand:')
                    print_card(hands)
                
                if env.get_perfect_information()['public_card']:
                    print("Community Cards:")
                    print_card(env.get_perfect_information()['public_card'])
                    # for hands in env.get_perfect_information()['public_card']:
                    #     print_card(hands)

                print('===============     Result     ===============')
                if payoffs[0] > 0:
                    print('You win {} chips!'.format(payoffs[0]))
                elif payoffs[0] == 0:
                    print('It is a tie.')
                else:
                    print('You lose {} chips!'.format(-payoffs[0]))
                print('')

                # evaluate convergence rate & plot reward values
                if i % convergence_interval == 0 and i != 0:
                    print("Evaluating...")
                    if not env.customAdversary:
                        logger.log_performance(
                            i,
                            tournament(
                                env,
                                convergence_interval,
                            )[0]
                        )
                    convergence_rate = env.agents[0].calculate_convergence_rate()
                    env.agents[0].update_target_network()
                    print(f'Convergence rate after {i} episodes: {convergence_rate}')

                if not is_training:
                    input("Press any key to continue...")

            # get paths
            csv_path, fig_path = logger.csv_path, logger.fig_path
        # plot the reward learning curve
        plot_curve(csv_path, fig_path, 'cfr')

    finally:
        if is_training:
            f.close()
