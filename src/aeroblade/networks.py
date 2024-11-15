import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            nn.init.eye_(self.input_layer.weight)
            nn.init.constant_(self.input_layer.bias, 0)

            nn.init.eye_(self.output_layer.weight)
            nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
                             
