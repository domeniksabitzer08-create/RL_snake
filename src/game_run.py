import pygame
import snake_classes
from snake_classes import *


pygame.init()
screen = pygame.display.set_mode((500, 500))


### DEBUG FUNCTIONS ###
def show_grid():
    for i in range(Grid.cell_count):
        for j in range(Grid.cell_count):
            cell = pygame.Rect((i * Grid.cell_size)+Grid.start_pos.x, (j* Grid.cell_size)+Grid.start_pos.y, Grid.cell_render_width, Grid.cell_render_width)
            pygame.draw.rect(screen, (200,0,0), cell)

### GAME FUNCTIONS ###
def reset():
    score = 0
    snake = snake_classes.SnakeManager(Vector2D(3, 4), Vector2D.right, 3)

### Timing Variables ###
clock = pygame.time.Clock()
current_time = pygame.time.get_ticks()
last_update = 0
update_interval = 250

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

### Start Game Logic ###
snake = snake_classes.SnakeManager(Vector2D(3,4),Vector2D.right, 3)

### UPDATE LOOP ###
while True:
    screen.fill((0, 0, 0))
    show_grid()
    # Call move_step depending on the update_interval
    current_time = pygame.time.get_ticks()
    if current_time - last_update >= update_interval:
        snake.move_step()
        last_update = current_time
    snake.handle_input()
    # Manage game over
    if snake.is_game_over:
        reset()
        snake = snake_classes.SnakeManager(Vector2D(3, 4), Vector2D.right, 3)

    snake.render_objects(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame.display.update()