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
    
    def train(self, input_data, target_data, num_epochs=1000, learning_rate=0.01, model_save_path=None):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            outputs = self(input_data)
            loss = criterion(outputs, target_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                # save model if specified
                if model_save_path:
                    torch.save(self.state_dict(), model_save_path)

    
    def predict(self, input_data):
        with torch.no_grad():
            return self(input_data)