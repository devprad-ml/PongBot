import pygame as py
import sys
import numpy as np
import torch
from dqn_agent import DQNAgent

py.init()

width , height = 1024, 768
screen = py.display.set_mode((width, height))
py.display.set_caption('Pong - Human vs Bot')
clock = py.time.Clock()

white = (255,255,255)
black = (0,0,0)

# paddle and the ball 

pad_width, pad_height = 7, 90
ball_size = 15

#positions of the paddles (py.Rect(x,y,width,height))
left_paddle = py.Rect(20,height//2 - pad_height//2, pad_width,pad_height)
right_paddle = py.Rect(width-30, height//2- pad_height//2, pad_width,pad_height)

#position of the ping pong ball

ball = py.Rect(width // 2, height // 2, ball_size, ball_size)
ball_speed_x = 4
ball_speed_y = 4

max_speed = 12

# font
score_font = py.font.SysFont('Papyrus', 40)

left_score, right_score = 0,0
score_limit = 10

# loading the AI agent
agent = DQNAgent(state_dim = 6, action_dim = 3)
agent.load_model("right_agent.pth")

# mode selection

mode = None

while mode not in ['1','2']:
    screen.fill(black)
    msg1 = score_font.render("Press 1 for single player", True, white)
    msg2 = score_font.render("Press 2 for two players", True, white)
    screen.blit(msg1,(width//2 -msg1.get_width()//2, height//2 -30))
    screen.blit(msg2, (width//2 - msg2.get_width()//2, height//2 + 30))
    py.display.flip()

    for event in py.event.get():
        if event.type == py.QUIT:
            py.quit()
            sys.exit()
        elif event.type == py.KEYDOWN:
            if event.key == py.K_1:
                mode = '1'
            elif event.key == py.K_2:
                mode = '2'

running = True
while running:
    screen.fill(black)
    for event in py.event.get():
        if event.type == py.QUIT:
            running = False
    
    keys = py.key.get_pressed()

    # left paddle(this will always be human)

    if keys[py.K_w] and left_paddle.top > 0:
        left_paddle.y = max(left_paddle.y - 6, 0)
    
    if keys[py.K_s] and left_paddle.bottom < height:
        left_paddle.y = min(left_paddle.y +6, height-pad_height)
    
    # right paddle (AI)

    if mode == '2':
        if keys[py.K_UP] and right_paddle.top >0:
            right_paddle.y = max(right_paddle.y -6,0)
        if keys[py.K_DOWN] and right_paddle.bottom < height:
            right_paddle.y = min(right_paddle.y + 6, height-pad_height)
        
    else:
        #uses DQN Agent
        state = np.array([
            left_paddle.y/height,
            right_paddle.y/height,
            ball.x/width,
            ball.y/height,
            ball_speed_x / max_speed,
            ball_speed_y / max_speed
        ])
        action = agent.select_action(state)
        if action == 1 and right_paddle.top > 0:
            right_paddle.y = max(right_paddle.y - 6, 0)
        elif action == 2 and right_paddle.bottom < height:
            right_paddle.y = min(right_paddle.y+6, height-pad_height)
    
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # bounce off the top and bottom width of the screen

    if ball.top <= 0 or ball.bottom >= height:
        ball_speed_y *= -1
    
    # paddle collision
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        ball_speed_x *= -1.1
        ball_speed_x = np.clip(ball_speed_x, -max_speed, max_speed)
    
    # score reset and update
    if ball.left <=0:
        right_score +=1
        ball.center = (width//2,height//2)
        ball_speed_x = np.random.choice([-4,4])
        ball_speed_y = np.random.choice([-2,2])
    elif ball.right>=width:
        left_score +=1
        ball.center = (width//2,height//2)
        ball_speed_x = np.random.choice([-4,4])
        ball_speed_y = np.random.choice([-2,2])
    
    # check for any wins
    if left_score >= score_limit or right_score >=score_limit:
        winner = "Left wins!" if left_score > right_score else "Right wins!"
        win_text = score_font.render(winner, True, white)
        screen.blit(win_text,(width//2-win_text.get_width()//2,height//2))
        py.display.flip()
        py.time.wait(3000)
        running = False
        continue

    py.draw.rect(screen, white, left_paddle)
    py.draw.rect(screen, white, right_paddle)
    py.draw.ellipse(screen, white, ball)
    
    # score text
    left_score_text = score_font.render(str(left_score),True,white)
    right_score_text = score_font.render(str(right_score),True,white)
    screen.blit(left_score_text,(width//4, 20))
    screen.blit(right_score_text, (3*width // 4, 20))

    py.display.flip()
    clock.tick(120)
    
py.quit()

        





        


