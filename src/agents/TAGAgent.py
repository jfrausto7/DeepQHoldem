from models.CardEvaluator.lookup import Lookup
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.ANN import ANN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TAGAgent(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, gamma=0.95):
        self.use_raw = False
        self.q_network = ANN(input_size, hidden_size, output_size)
        self.target_network = ANN(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.lookup = Lookup()
        self.low_threshold = -100
        self.high_threshold = 100
        self.bluffing_rate = 0.2

        # make accessible to GPU
        self.q_network.to(device)
        self.target_network.to(device)
        
    def card_transform(self, cards):
      '''
      Format conversion for lookup table
      Example: ['CA', 'S2'] -> ['Ac', '2s']
      '''
      if len(cards) == 0:
          return cards
      cards_T = []
      for card in cards:
          card = card[1] + card[0].lower()
          cards_T.append(card)
      return cards_T

    def evaluate_hand_strength(self, obs):
        hand_cards = self.card_transform(obs['raw_obs']['hand'])
        community_cards = self.card_transform(obs['raw_obs']['public_cards'])
        return self.lookup.calc(hand_cards, community_cards)

    def evaluate_weight_for_action(self, action, hand_strength, raise_actions):
        if action in raise_actions:
            return hand_strength / 100.0
        else:
            return 1.0
        
    def step(self, state):
      # tight-aggressive strategy for evaluation
      legal_actions = list(state['legal_actions'].keys())
      actions = np.zeros(list(self.q_network.modules())[-1].out_features)
      actions[legal_actions] = 1
      s = torch.from_numpy(np.concatenate((state['obs'], actions)).astype(np.float32))
      s = s.to(device)

      # evaluate hand strength
      hand_strength = self.evaluate_hand_strength(state)

      # play best action (highest Q-value) based on hand strength if available, otherwise, check/fold
      if hand_strength < self.low_threshold:
          # fold or bluff
          rand_num = np.random.uniform(0, 1)
          raise_actions = [a for a in legal_actions if a > 1 and a < len(legal_actions) - 1]
          if (rand_num < self.bluffing_rate) and raise_actions:
              action = np.random.choice(raise_actions)
              return legal_actions[action]

          return legal_actions[0]  # fold
      elif hand_strength < self.high_threshold:
          if 1 in legal_actions:
              return legal_actions[1]  # check/call
          else:
              return legal_actions[0]  # fold
      else:
          # play a fractional raise action with the highest Q-value based on hand strength if available
          raise_actions = [a for a in legal_actions if a > 1 and a < len(legal_actions) - 1]
          if raise_actions:
              q_values = self.q_network(s)
              weighted_q_values = [q_values[a] * self.evaluate_weight_for_action(a, hand_strength, raise_actions) for a in raise_actions]
              max_raise_index = raise_actions[int(torch.argmax(torch.tensor(weighted_q_values)).detach().numpy())]
              return legal_actions[max_raise_index]
          else:
              if 1 in legal_actions:
                  return legal_actions[1]  # check
              else:
                  return legal_actions[0]  # fold
        

    def eval_step(self, state):
      return self.step(state), {}

    def select_action(self, state, epsilon, legal_actions):
        if random.random() < epsilon:
            return torch.tensor([random.randrange(list(self.q_network.modules())[-1].out_features)], dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)

                # filter Q-values for legal actions
                legal_q_values = q_values[legal_actions]
                max_index = legal_actions[torch.argmax(legal_q_values)]
                return torch.tensor([max_index], dtype=torch.long)
