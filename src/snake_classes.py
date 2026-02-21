import pygame
from dataclasses import dataclass
import time
from random import randint

### VECTOR CLASS ###

@dataclass(frozen=True)
class Vector2D:
        x: float
        y: float
        def __add__(self, other):
            # Vector + Vector
            if isinstance(other, Vector2D):
                return Vector2D(self.x + other.x, self.y + other.y)
        def __sub__(self, other):
            # Vector - Vector
            if isinstance(other, Vector2D):
                return Vector2D(self.x - other.x, self.y - other.y)
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                # Vector * Number
                return Vector2D(self.x * other, self.y * other)
        def __truediv__(self, other):
            # Vector / Number
            if isinstance(other, (int, float)):
                return Vector2D(self.x / other, self.y / other)


# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

class Grid:
    start_pos = Vector2D(100, 100)
    cell_count = 10
    cell_size = 30
    cell_render_width = 10

# Function to convert grid to screen and screen to grid pos
def screen_to_grid_pos(pos: Vector2D):
    """Converts the screen position to the grid position of the game field"""
    nearest_pos_x = int((pos.x - Grid.start_pos.x)/ Grid.cell_size)
    nearest_pos_y = int((pos.y - Grid.start_pos.y) / Grid.cell_size)
    return Vector2D(nearest_pos_x, nearest_pos_y)

def grid_to_screen_pos(pos: Vector2D):
    """Converts the grid position to the screen position"""
    x = (pos.x * Grid.cell_size) + Grid.start_pos.x
    y = (pos.y * Grid.cell_size) + Grid.start_pos.y
    return Vector2D(x, y)

### Design Variables ###

### Snake
part_color = (0,255,0)
# How much the color decreases over time
part_color_reduction_rate = 30
# Size of the parts
snake_part_render_size = 25

### Food
food_color = (255,0,0)
food_render_size = 30

class SnakePart:
    def __init__(self, part_list: iter):
        self.index = len(part_list)
        # If the part is the first part, there is no pre_part
        if not self.index == 0:
            self.pre_part = part_list[self.index-1]
        else:
            self.pre_part = None
        self.pos = None
    # Movement functions
    def move_to_direction(self, direction: Vector2D):
        """Moves the part in a certain direction."""
        self.pos += direction

    def move_to_pre_part(self):
        """Moves to the grid position of the predecessor."""
        self.pos = self.pre_part.pos


    # Render functions
    def render(self, screen: pygame.Surface):
        screen_pos = grid_to_screen_pos(self.pos)
        part = pygame.Rect(screen_pos.x, screen_pos.y, snake_part_render_size, snake_part_render_size)
        new_color = (part_color[0], part_color[1] -self.index * part_color_reduction_rate, part_color[2])
        pygame.draw.rect(screen, new_color, part)

class Food:
    def __init__(self, pos: Vector2D):
        self.pos = pos
    def render(self, screen: pygame.Surface):
        food = pygame.Rect(0, 0, food_render_size, food_render_size)
        pygame.draw.rect(screen, food_color, food)

class SnakeManager:
    """Manages spawning and moving of the parts"""
    def __init__(self, start_pos: Vector2D, start_direction: Vector2D, n_starting_parts: int):
        self.start_pos = start_pos
        self.start_direction = start_direction
        self.n_starting_parts = n_starting_parts
        self.part_list = []
        self.direction = start_direction
        self.is_dir_changing = False
        self.init_parts()

    def init_parts(self):
        # Init first part
        self.part_list.append(SnakePart(self.part_list))
        self.part_list[0].pos = self.start_pos
        # Init the other parts
        for i in range(self.n_starting_parts-1):
            self.part_list.append(SnakePart(self.part_list))
            self.part_list[i + 1].pos = Vector2D(self.start_pos.x +  ((i+1) * self.start_direction.x*-1),
                                                 self.start_pos.y + ((i+1) * self.start_direction.y*-1))

    def move_step(self):
        time.sleep(0.3)

        # Move each part to the position of the part before
        for i in range(len(self.part_list),1,-1):
            print(f"i:{i} | len: {len(self.part_list)}")
            self.part_list[i-1].move_to_pre_part()
        # Move the first part
        self.part_list[0].move_to_direction(self.direction)
        self.is_dir_changing = False


    def change_direction(self, new_direction: Vector2D):
        if not (new_direction*-1) == self.direction and not self.is_dir_changing:
            self.is_dir_changing = True
            self.direction = new_direction

    def handle_input(self):
        if pygame.key.get_pressed()[pygame.K_w]:
            self.change_direction(Vector2D.up)
        elif pygame.key.get_pressed()[pygame.K_s]:
            self.change_direction(Vector2D.down)
        elif pygame.key.get_pressed()[pygame.K_d]:
            self.change_direction(Vector2D.right)
        elif pygame.key.get_pressed()[pygame.K_a]:
            self.change_direction(Vector2D.left)


    def render_parts(self, screen: pygame.Surface):
        # Render all parts
        for part in self.part_list:
            part.render(screen)




