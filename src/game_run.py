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
    snake.move_step()
    #handel_input()


    snake.render_parts(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame.display.update()