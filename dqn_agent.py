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

        # replay buffer

        self.replay_buffer = deque(maxlen = 50000)
        self.batch_size = 64
    
    # the actual learning will happen here. 

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # get random batch from memory

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # converting parameters into tensors

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # get current Q values
        q_values = self.model(states).gather(1, actions)

        # calc. target Q values using Bellman equations

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1, keepdim = True)[0]
            target_q_values = rewards + (1-dones) * self.gamma * next_q_values

        # calc. loss and update network

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon after each training step

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
    
    def save_model(self, path):
        torch.save(self.model.state.dict(), path)

    
    # load model weights from file

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location = self.device))
        self.model.eval()



        
        
        
