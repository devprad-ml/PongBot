''' this is the environment created for training the bot. 
The environment will be created using OpenAI gym.
The bot will be trained by playing against itself. 
Agent: the paddle.
State: It will be a continuous state as the agent will have to move according to the ball.
Reward : +1 to hit the ball, -5 if it loses, +5 if it wins.'''

import numpy as np
import random
# we are defining the objects and constants that will be present in the environment at all times
class PongEnvironment:
    def __init__(self, width = 1024, height = 768, pad_h = 90, pad_w = 5, ball_rad = 5):
        self.width = width
        self.height = height
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.ball_rad = ball_rad
        self.max_speed = 12
        self.score_limit = 10    # pts needed to win a game
        self.reset()

    # func. to reset the environment
    
    def reset(self):
        
        # resetting the paddles 
        self.left_y = self.height //2 - self.pad_h//2
        self.right_y = self.height //2 - self.pad_h//2
        # resetting the pos. of the ball and randomizing the ball movement
        self.ball_y = self.height //2
        self.ball_x = self.height //2
        self.ball_vx = np.random.choice([-4,4])
        self.ball_vy = np.random.choice([-2,2])

        # score
        self.left_score = 0
        self.right_score = 0
        self.done = False
        return self.get_state()
    
    def reset_ball(self):
        self.ball_x = self.width//2
        self.ball_y = self.height // 2
        self.ball_vx = np.random.choice([-4,4])
        self.ball_vy = np.random.choice([-2,2])
    
    # retrieving the state in an array

    def get_state(self):
        return np.array([
            # storing the positions of the paddles
            self.left_y/self.height,
            self.right_y/ self.height,
            # storing positions of the ball in x-y co-ordinates
            self.ball_x / self.width,
            self.ball_y / self.height,
            # storing velocity of the ball
            self.ball_vx/ self.max_speed,
            self.ball_vy/self.max_speed 
        ])
 # defining everything about the actions   
    def step(self, action_left, action_right):
        reward_left = 0
        reward_right = 0
        # defining paddle speed
        dy = 6

        # making the left paddle move up and down
        if action_left ==1 and self.left_y > 0:
            self.left_y -= dy
        
        elif action_left == 2 and self.left_y + self.pad_h < self.height:
            self.left_y += dy

        # making the right paddle move up and down

        if action_right ==1 and self.right_y > 0:
            self.right_y -= dy
        elif action_right == 2 and self.right_y + self.pad_h < self.height: # this line ensures the paddle does not go off screen
            self.right_y += dy
        
        # making the ball move

        self.ball_y += self.ball_vy
        self.ball_x += self.ball_vx

        # bounce off the top and bottom

        if self.ball_y <= 0 and self.ball_y >= self.height:
            self.ball_vy *= -1
        
        # ball and paddle collisions
        
        if (self.ball_x <=10 + self.pad_w  # ball is near horizontal area of the paddle
            and
            self.left_y <= self.ball_y <=self.left_y + self.pad_h):  # ball is in between paddle
            self.ball_vx *= -1.1
            reward_left += 1
        
        elif(self.ball_x >= self.width -10 - self.pad_w 
             and 
             self.right_y <=self.ball_y <=self.right_y + self.pad_h
             ):
            self.ball_vx *= -1.1
            reward_right += 1
        
        # capping max speed
        self.ball_vx = np.clip(self.ball_vx, -self.max_speed, self.max_speed)
        self.ball_vy = np.clip(self.ball_vy, -self.max_speed, self.max_speed)


        # scoring

        
        #right wins
        if self.ball_x < 0:
            self.right_score +=1
            reward_right = 5
            reward_left = -5
            self.reset_ball()
            # left wins
        elif self.ball_x > self.width:
            reward_left = 5
            reward_right = -5
            self.reset_ball()
        
        if self.left_score >= self.score_limit or self.right_score >= self.score_limit:
            self.done = True
        
        return self.get_state(), reward_left, reward_right, self.done
        


        


    

    



