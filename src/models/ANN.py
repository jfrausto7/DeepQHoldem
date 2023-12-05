import torch
import torch.nn as nn
import torch.optim as optim

# ANN architecture (loosely inspired by: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch)
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.layer_3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.leaky_relu_1(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.leaky_relu_2(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)
        return x