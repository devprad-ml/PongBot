import pygame
import sys

# Initializing the game GUI
pygame.init()

# Constants
width, height = 800, 600
white = (255, 255, 255)
red = (255,0,0)
blue = (0,0,255)
black = (0, 0, 0)

#setting up the paddle widht and height and also the ball radius
pad_width, pad_height = 5, 75
ball_rad = 5

# Setup
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong")
clock = pygame.time.Clock()

# positions of the left and right paddle and the ball
left_paddle = pygame.Rect(10, height//2 - pad_height//2, pad_width, pad_height)
right_paddle = pygame.Rect(width - 20, height//2 - pad_height//2, pad_width, pad_height)
ball = pygame.Rect(width//2, height//2, ball_rad*2, ball_rad*2)

# this is so that the game has delay and the ball does not start moving instantly
started = False



# Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(black)

    # Draw paddles and ball
    pygame.draw.rect(screen, red, left_paddle)
    pygame.draw.rect(screen, blue, right_paddle)
    pygame.draw.ellipse(screen, white, ball)

    pygame.display.flip()
    clock.tick(120)
