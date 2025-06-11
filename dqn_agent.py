'''We will create a training regiment for AI agents that will play against
themselves to improve using Neural Networks.'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# Creating the Neural Network from scratch

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self,x):
        return self.model(x)

# creating the Deep Q Network agent 
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon= 1.0, epsilon_decay = 0.995,min_epsilon = 0.05):
        # environment dimensions
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'):

        # Q-Network

        self.model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()  # minimizes Temporal Difference error

        
        
