import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pong_env import PongEnvironment
from dqn_agent import DQNAgent
import os
import csv

left_model_path = 'left_agent.pth'
right_model_path = 'right_agent.pth'
meta_path = 'meta.txt'

# saving checkpoint to not lose progress
def save_checkpoint(left_agent, right_agent, episode):
    left_agent.save_model(left_model_path)
    right_agent.save_model(right_model_path)
    with open(meta_path, "w") as f:

        f.write(str(episode))
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(left_agent, right_agent):

    start_episode = 0
    if os.path.exists(left_model_path) and os.path.exists(right_model_path):
        left_agent.load_model(left_model_path)
        right_agent.load_model(right_model_path)
        print("Models loaded from checkpoint.")
    
    #resuming training from this checkpoint

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            start_episode = int(f.read())
        print(f"Resuming from episode {start_episode}")
    return start_episode

    
     
    
    


def train(num_episodes = 1000):
# initializing the pong environment class
    env = PongEnvironment()
    state_dim = 6 
    action_dim = 3

    # initializing the agents
    left_agent = DQNAgent(state_dim,action_dim)
    right_agent = DQNAgent(state_dim,action_dim)

    # resume if the checkpoint exists

    start_episode = load_checkpoint(left_agent, right_agent)
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/training_log.csv"
    write_header = not os.path.exists(log_path)

    log_file = open(log_path, mode='a', newline='')
    csv_writer = csv.writer(log_file)

    if write_header:
        csv_writer.writerow([
            "Episode", "Reward_Left", "Reward_Right",
            "Epsilon_Left", "Epsilon_Right",
            "Loss_Left", "Loss_Right",
            "MaxQ_Left", "MaxQ_Right",
            "Outcome"
        ])

    # getting initial state
    try:


        for episode in range(start_episode,num_episodes):
            state = env.reset()
            total_reward_left = 0
            total_reward_right = 0
            done = False

            losses_left = []
            losses_right = []
            qvals_left = []
            qvals_right = []

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
                # print the episodes to check progress
            print(f"Episode{episode+1}/{num_episodes} - Left: {total_reward_left:.2f}, Right: {total_reward_right:.2f}")

            avg_loss_left = np.mean(losses_left) if losses_left else 0.0
            avg_loss_right = np.mean(losses_right) if losses_right else 0.0
            avg_q_left = np.mean(qvals_left) if qvals_left else 0.0
            avg_q_right = np.mean(qvals_right) if qvals_right else 0.0

            # Determine outcome
            if total_reward_left > total_reward_right:
                outcome = "Left Wins"
            elif total_reward_left < total_reward_right:
                outcome = "Right Wins"
            else:
                outcome = "Draw"


            csv_writer.writerow([
                episode + 1,
                total_reward_left, total_reward_right,
                left_agent.epsilon, right_agent.epsilon,
                avg_loss_left, avg_loss_right,
                avg_q_left, avg_q_right,
                outcome
])


            

    except KeyboardInterrupt:
            print('\nTraining interrupted. Saving checkpoint...')
            save_checkpoint(left_agent,right_agent, episode+1)
    
    finally:
        left_agent.save_model("left_agent.pth")
        right_agent.save_model("right_agent.pth")
        print("Models saved successfully")
            
            
    
    
        

if __name__ == "__main__":
    train()