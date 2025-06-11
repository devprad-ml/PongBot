import pygame as py
import sys

# Initializing the game GUI
py.init()

# Constants
width, height = 800, 600
white = (255, 255, 255)
red = (255,0,0)
blue = (0,0,255)
black = (0, 0, 0)

#setting up the paddle width and height and also the ball radius
pad_width, pad_height = 5, 75
ball_rad = 5

ball_speed_x = 4
ball_speed_y = 4


# setting the movement speed of the ball in x and y direction

# Setup
screen = py.display.set_mode((width, height))
py.display.set_caption("Pong")
clock = py.time.Clock()

# positions of the left and right paddle and the ball
left_paddle = py.Rect(10, height//2 - pad_height//2, pad_width, pad_height)
right_paddle = py.Rect(width - 20, height//2 - pad_height//2, pad_width, pad_height)
ball = py.Rect(width//2, height//2, ball_rad*2, ball_rad*2)

# this is so that the game has delay and the ball does not start moving instantly
started = True
font = py.font.SysFont(None, 60)

while started:
    screen.fill(black)
    msg = font.render("Press Enter to start!", True, white)
    screen.blit(msg, (width//2 - msg.get_width()//2, height//2))
    py.display.flip()

    for event in py.event.get():
        if event.type == py.QUIT:
            py.quit()
            sys.exit()
        elif event.type == py.KEYDOWN:
            if event.key == py.K_RETURN:
                started = False

for i in range(3,0,-1):
    screen.fill(black)
    countdown = font.render(str(i),True, white)
    screen.blit(countdown, (width//2 - countdown.get_width( )//2, height//2))
    py.display.flip()
    py.time.delay(1000)

# Keeping count of scores in these variables.
left_score = 0
right_score = 0
score_font = py.font.SysFont(None, 35)

# Game Loop
while True:
    for event in py.event.get():
        if event.type == py.QUIT:
            py.quit()
            sys.exit()
        
    ball.x += ball_speed_x     # ball movement
    ball.y += ball_speed_y

    if ball.top <=0 or ball.bottom >= height:   # ball bounce 
        ball_speed_y *= -1 
    
    # right paddle scores a point
    if ball.left <=0:
        right_score +=1
        ball.x,ball.y = width//2,height//2  # return ball to center
        ball_speed_x *= -1
    
    # left paddle scores a point
    if ball.right >= width:
        left_score +=1
        ball.x,ball.y = width//2,height//2 
        ball_speed_x *= -1

    # now let us add paddle movement

    keys = py.key.get_pressed()

    if keys[py.K_w] and left_paddle.top > 0:   # up
        left_paddle.y -=6
    
    if keys[py.K_s] and left_paddle.top < height:   # down
        left_paddle.y +=6
    
    if keys[py.K_UP] and right_paddle.top > 0:
        right_paddle.y -=6
    
    if keys[py.K_DOWN] and right_paddle.top < height:
        right_paddle.y +=6

    
    # paddle collision mechanics

    if ball.colliderect(left_paddle) and ball_speed_x <0:
        ball_speed_x *= -1.1
    
    if ball.colliderect(right_paddle) and ball_speed_x > 0:
        ball_speed_x *= -1.1
    
    max_speed = 6

    ball_speed_x = max(-max_speed, min(ball_speed_x, max_speed))
    ball_speed_y = max(-max_speed, min(ball_speed_y, max_speed))

    screen.fill(black)

    # Draw paddles and ball
    py.draw.rect(screen, red, left_paddle)
    py.draw.rect(screen, blue, right_paddle)
    py.draw.ellipse(screen, white, ball)

    left_text = score_font.render(str(left_score), True, red)
    right_text = score_font.render(str(right_score), True, blue)
    screen.blit(left_text,(width//4, 20))
    screen.blit(right_text, (width *3 // 4,20))

    py.display.flip()
    clock.tick(120)
