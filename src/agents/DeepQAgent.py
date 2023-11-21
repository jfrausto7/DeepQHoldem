import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from models.ANN import ANN

'''Simplified implementation of agent described in "Human-level control through deep reinforcement
 learning" by Mnih et al. (https://www.nature.com/articles/nature14236)
'''
class DeepQAgent(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, gamma=0.95):
        self.use_raw = False
        self.q_network = ANN(input_size, hidden_size, output_size)
        self.target_network = ANN(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.convergence_rates = []
    
    def step(self, state):
        legal_actions = list(state['legal_actions'].keys())
        actions = np.zeros(list(self.q_network.modules())[-1].out_features)
        actions[legal_actions] = 1
        s = torch.from_numpy(np.concatenate((state['obs'], actions)).astype(np.float32))

        action = self.select_action(s, 0.1).item()
        while action not in legal_actions:
            action = self.select_action(s, 0.1).item()

        return action
    
    def eval_step(self, state):
        return self.step(state), {}

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return torch.tensor([random.randrange(list(self.q_network.modules())[-1].out_features)], dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values)

    def train(self, state, action, next_state, reward):
        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)

        target_q_values = q_values.clone()
        target_q_values[action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_convergence_rate(self):
        q_network_params = list(self.q_network.parameters())
        target_network_params = list(self.target_network.parameters())
        current_convergence_rates = []

        for q_param, target_param in zip(q_network_params, target_network_params):
            q_param_data = q_param.data.flatten()
            target_param_data = target_param.data.flatten()

            convergence_rate = torch.mean(torch.abs(q_param_data - target_param_data)).item()
            current_convergence_rates.append(convergence_rate)

        self.convergence_rates.append(current_convergence_rates)
        return current_convergence_rates


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
