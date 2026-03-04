import time

import pygame
import snake_classes
from snake_classes import *


pygame.init()
screen = pygame.display.set_mode((500, 500))




### GAME FUNCTIONS ###
def reset():
    score = 0
    snake = snake_classes.SnakeManager(Vector2D(3, 4), Vector2D.right, 3)

### Timing Variables ###
clock = pygame.time.Clock()
current_time = pygame.time.get_ticks()
last_update = 0
update_interval = 600

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

### Start Game Logic ###
snake = snake_classes.SnakeManager(Vector2D(0,0),Vector2D.right, 3)

### UPDATE LOOP ###

state, reward, is_game_over, score = snake.step([0,0,1])
print(f"current state: {state}")
while True:
    time.sleep(4)
    state = snake.reset()
    time.sleep(10)
    # Call move_step depending on the update_interval
    current_time = pygame.time.get_ticks()
    if current_time - last_update >= update_interval:
        last_update = current_time
    # Manage game over
    if snake.is_game_over:
        reset()
        snake = snake_classes.SnakeManager(Vector2D(3, 4), Vector2D.right, 3)

    snake.render_objects(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame.display.update()

