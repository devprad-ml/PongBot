import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pong_env import PongEnvironment
from dqn_agent import DQNAgent

def train(num_episodes = 10):


# initializing the pong environment class
    env = PongEnvironment()
    state_dim = 6 
    action_dim = 3

    # initializing the agents
    left_agent = DQNAgent(state_dim,action_dim)
    right_agent = DQNAgent(state_dim,action_dim)

    # getting initial state

    for episode in range(num_episodes):
        state = env.reset()
        total_reward_left = 0
        total_reward_right = 0
        done = False

        while not done:
            action_left = left_agent.select_action(state)
            action_right = right_agent.select_action(state)

            # make the environment go forward, i.e. actually play now

            next_state, reward_left, reward_right, done = env.step(action_left, action_right)


            # store transitions in each agents memory

            left_agent.store_transition(state, action_left, reward_left, next_state, done)
            right_agent.store_transition(state, action_right, reward_right, next_state, done)

            # learn from memory stored by above code

            left_agent.train()
            right_agent.train()

            state = next_state
            total_reward_left += reward_left
            total_reward_right += reward_right
        
        print(f"Episode{episode+1}/{num_episodes} - Left: {total_reward_left:.2f}, Right: {total_reward_right:.2f}")
    
    left_agent.save_model("left_agent.pth")
    right_agent.save_model("right_agent.pth")
        

if __name__ == "__main__":
    train()