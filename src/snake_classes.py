

import pygame
from dataclasses import dataclass
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
            else:
                return NotImplemented
        def __sub__(self, other):
            # Vector - Vector
            if isinstance(other, Vector2D):
                return Vector2D(self.x - other.x, self.y - other.y)
            else:
                return NotImplemented
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                # Vector * Number
                return Vector2D(self.x * other, self.y * other)
            else:
                return NotImplemented
        def __truediv__(self, other):
            # Vector / Number
            if isinstance(other, (int, float)):
                return Vector2D(self.x / other, self.y / other)
            else:
                return NotImplemented

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
part_color_reduction_rate = 5
# Size of the parts
snake_part_render_size = 25

### Food
food_color = (255,0,0)
food_render_size = 30

class SnakeManager:
    """Manages spawning and moving of the parts"""
    def __init__(self, start_pos: Vector2D, start_direction: Vector2D, n_starting_parts: int):
        # Snake
        self.start_pos = start_pos
        self.start_direction = start_direction
        self.n_starting_parts = n_starting_parts
        self.part_list = []
        self.direction = start_direction
        self.is_dir_changing = False
        # Food
        self.food = None
        self.init_parts()
        # Other
        self.is_game_over = False

    def init_parts(self):
        """Init all parts and the first food"""
        # Init first part
        self.part_list.append(self.start_pos) # First part
        # Init the other parts
        for i in range(self.n_starting_parts-1):
            self.part_list.append(Vector2D(self.start_pos.x +  ((i+1) * self.start_direction.x*-1),
                                           self.start_pos.y + ((i+1) * self.start_direction.y*-1)))
        # Init food
        self.food = self.init_food()

    def move_step(self):
        # Move each part to the position of the part before, but not the first one
        for i in range(len(self.part_list),1,-1):
            self.part_list[i-1] = self.part_list[i-2]
        # Move the first part
        self.part_list[0] += self.direction
        self.is_dir_changing = False

        ### Check Collisions ##
        # Check if head collides with food
        self.check_food_collision()
        # Check if Snake collides with other part of snake
        if self.check_other_part_collision(self.part_list[0]):
            self.game_over()
        # Check if snake head collides with border
        self.check_border_collision()


    def check_food_collision(self):
        """Adds a Part if head collides with food and init a new food"""
        if self.part_list[0] == self.food:
            self.add_part()
            self.food = self.init_food()

    def init_food(self) -> Vector2D:
        """Init on a random pos and returns the food object"""
        # Create one instance of Food on rnd pos
        spawn_pos = Vector2D(randint(0,Grid.cell_count-1), randint(0,Grid.cell_count-1))
        # If it collides with a part it will try again
        while self.check_other_part_collision(spawn_pos):
            spawn_pos = Vector2D(randint(0, Grid.cell_count-1), randint(0, Grid.cell_count-1))
        return spawn_pos

    def add_part(self):
        """Init a new part in the next tick"""
        self.part_list.append(self.part_list[len(self.part_list)-1])

    def check_other_part_collision(self, obj_pos: Vector2D) -> bool:
        """Returns True if the object collides with any other part of the snake."""
        for i in range(1,len(self.part_list)):
            if self.part_list[i] == obj_pos:
                return True
        return False

    def check_border_collision(self):
        x = self.part_list[0].x
        y = self.part_list[0].y
        if  x > Grid.cell_count-1 or x < 0 :
            self.game_over()
        if y > Grid.cell_count-1 or y < 0 :
            self.game_over()

    def change_direction(self, new_direction: Vector2D):
        if not (new_direction*-1) == self.direction and not self.is_dir_changing:
            self.is_dir_changing = True
            self.direction = new_direction

    def game_over(self):
        print("Game Over")
        self.is_game_over = True

    def handle_input(self):
        if pygame.key.get_pressed()[pygame.K_w]:
            self.change_direction(Vector2D.up)
        elif pygame.key.get_pressed()[pygame.K_s]:
            self.change_direction(Vector2D.down)
        elif pygame.key.get_pressed()[pygame.K_d]:
            self.change_direction(Vector2D.right)
        elif pygame.key.get_pressed()[pygame.K_a]:
            self.change_direction(Vector2D.left)


    def render_objects(self, screen: pygame.Surface):
        # Render all parts
        self.render_parts(screen)
        # Render food
        self.render_food(screen)

    def render_parts(self, screen: pygame.Surface):
        """Render all snake parts"""
        for part_pos in self.part_list:
            screen_pos = grid_to_screen_pos(part_pos)
            part = pygame.Rect(screen_pos.x, screen_pos.y, snake_part_render_size, snake_part_render_size)
            new_color = (part_color[0], part_color[1] , part_color[2])
            pygame.draw.rect(screen, new_color, part)

    def render_food(self, screen: pygame.Surface):
        screen_pos = grid_to_screen_pos(self.food)
        food = pygame.Rect(screen_pos.x, screen_pos.y, food_render_size, food_render_size)
        pygame.draw.rect(screen, food_color, food)



