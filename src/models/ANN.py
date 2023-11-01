import torch
import torch.nn as nn
import torch.optim as optim

# ANN architecture (loosely inspired by: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch)
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x