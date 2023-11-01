import torch
import torch.nn as nn
import torch.optim as optim
import random

from models.ANN import ANN

'''Simplified implementation of agent described in "Human-level control through deep reinforcement
 learning" by Mnih et al. (https://www.nature.com/articles/nature14236)
'''
class DeepQAgent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, gamma=0.99):
        self.q_network = ANN(input_size, hidden_size, output_size)
        self.target_network = ANN(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return torch.tensor([random.randrange(self.q_network.out_features)], dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values)

    def train(self, state, action, next_state, reward, done):
        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)
        target_q_values = q_values.clone()

        if done:
            target_q_values[0][action] = reward
        else:
            target_q_values[0][action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
