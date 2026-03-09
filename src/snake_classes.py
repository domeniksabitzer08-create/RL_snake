
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
part_color = (0,150,0)
# How much the color decreases over time
part_color_reduction_rate = 5
# Size of the parts
snake_part_render_size = 25

### Food
food_color = (255,0,0)
food_render_size = 30

### REINFORCEMENT LEARNING VARIABLES ###
NOTHING_REWARD = 0
EAT_FOOD_REWARD = 10
GAME_OVER_REWARD = -10

class SnakeManager:
    """Manages spawning and moving of the parts"""
    def __init__(self, start_pos: Vector2D, start_direction: Vector2D, n_starting_parts: int, render: bool = True):
        # rendering
        self.render = render
        # Snake
        self.start_pos = start_pos
        self.start_direction = start_direction
        self.start_direction = start_direction
        self.n_starting_parts = n_starting_parts
        self.part_list = []
        self.direction = start_direction
        self.is_dir_changing = False
        # available directions      ----right-----------down----------left----------up------
        self.available_directions = [Vector2D(1,0),Vector2D(0,1),Vector2D(-1,0),Vector2D(0,-1)]
        # Food
        self.food = None
        # init parts
        self.init_parts()
        # Reinforcement Learning
        self.reward = 0
        #self.observation_space
        # Other
        self.is_game_over = False

        self.score = 0

        # Setup PyGame if it needs to be rendered
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((500, 500))



    def init_parts(self):
        """Init all parts and the first food"""
        self.part_list = []
        # Init first part
        self.part_list.append(self.start_pos) # First part
        # Init the other parts
        for i in range(self.n_starting_parts-1):
            self.part_list.append(Vector2D(self.start_pos.x +  ((i+1) * self.start_direction.x*-1),
                                           self.start_pos.y + ((i+1) * self.start_direction.y*-1)))
        # Init food
        self.food = self.init_food()

    def reset(self):
        # Snake
        self.part_list = []
        self.direction = self.start_direction
        self.is_dir_changing = False
        # Food
        self.food = None
        # init parts
        self.init_parts()
        # Reinforcement Learning
        self.reward = 0
        # Other
        self.is_game_over = False
        self.score = 0

        self.init_parts()
        state = self.get_state()
        return state

    def step(self, action):
        # render if necessary
        if self.render:
            self.render_objects(self.screen)
            pygame.display.update()
        self.change_direction(action)
        self.reward = NOTHING_REWARD
        # Move each part to the position of the part before, but not the first one
        for i in range(len(self.part_list),1,-1):
            self.part_list[i-1] = self.part_list[i-2]
        # Move the first part
        self.part_list[0] += self.direction
        self.is_dir_changing = False

        ### RL ONLY ###

        # get the state
        state = self.get_state()
        ### Check Collisions ##
        # Check if head collides with food
        self.check_food_collision()
        # Check if Snake collides with other part of snake
        if self.check_other_part_collision(self.part_list[0]):
            self.game_over()
        # Check if snake head collides with border
        if self.check_border_collision(self.part_list[0]):
            self.game_over()
        self.score = len(self.part_list) - self.n_starting_parts

        return state, self.reward, self.is_game_over, self.score


    def check_food_collision(self):
        """Adds a Part if head collides with food and init a new food"""
        if self.part_list[0] == self.food:
            self.reward = EAT_FOOD_REWARD
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

    def check_border_collision(self, part_pos: Vector2D ) -> bool:
        """returns true if pos is outside the border"""
        x = part_pos.x
        y = part_pos.y
        if  x > Grid.cell_count-1 or x < 0 :
            return True
        if y > Grid.cell_count-1 or y < 0 :
            return True
        else:
            return False

    def change_direction(self, action):
        # 0 -> left
        # 1 -> straight
        # 2 -> right
        clockwise_dir = [Vector2D.right, Vector2D.down, Vector2D.left, Vector2D.up]
        idx = clockwise_dir.index(self.direction) # the direction as index in clockwise array
        # 1. Make a left turn
        if action == 0:
            new_idx = idx-1
            if new_idx == -1:
                new_idx = 3
        # 2. Stay straight
        elif action == 1:
            new_idx = idx
        # 3. Make a right turn
        else:
            new_idx = idx+1
            if new_idx == 4:
                new_idx = 0
        # Set the new direction depending on the action
        self.direction = clockwise_dir[new_idx]

    def game_over(self):
        self.reward = GAME_OVER_REWARD
        self.is_game_over = True

    def sample(self):
        """returns a random action"""
        return randint(0, 2)


    # only for RL
    def get_state(self):
        """
        returns the current state.
        """
        clockwise_dir = [Vector2D.right, Vector2D.down, Vector2D.left, Vector2D.up]
        # create array danger[0,0,1] then get idx and this for every state
        danger = self.get_danger(self.part_list[0])
        # get direction as int
        direction_idx = clockwise_dir.index(self.direction)
        food_dir = self.get_food_dir()

        danger_int = self.convert_to_int(danger)
        food_dir_int = self.convert_to_int(food_dir)


        # create state
        state = (direction_idx * (8*16) + danger_int * 16 + food_dir_int)
        return state

    def get_danger(self, state_pos):
        # Danger [Left, Front, Right]
        clockwise_dir = [Vector2D.right, Vector2D.down, Vector2D.left, Vector2D.up]
        danger = [0,0,0]

        front_idx = clockwise_dir.index(self.direction)
        front_dir = clockwise_dir[front_idx]
        front = state_pos + front_dir

        left_idx = front_idx -1
        # if the idx is -1, then the next direction would be up (idx=3)
        if left_idx == -1:
            left_idx = 3
        left_dir = clockwise_dir[left_idx]
        left = state_pos + left_dir

        right_idx = front_idx + 1
        if right_idx == 4:
            right_idx = 0
        right_dir = clockwise_dir[right_idx]
        right = state_pos + right_dir

        # Check in front
        if self.check_border_collision(front) or self.check_other_part_collision(front):
            danger[1] = 1
        else:
            danger[1] = 0
        # Check left
        if self.check_border_collision(left) or self.check_other_part_collision(left):
            danger[0] = 1
        else:
            danger[0] = 0
        # Check right
        if self.check_border_collision(right) or self.check_other_part_collision(right):
            danger[2] = 1
        else:
            danger[2] = 0

        return danger

    def get_food_dir(self):
        food = self.food
        head = self.part_list[0]
        dx = food.x - head.x
        dy = food.y - head.y
        if self.direction == Vector2D.up:
            food_left = dx < 0
            food_right = dx > 0
            food_front = dy < 0
            food_back = dy > 0
        elif self.direction == Vector2D.right:
            food_left = dy < 0
            food_right = dy > 0
            food_front = dx > 0
            food_back = dx < 0
        elif self.direction == Vector2D.down:
            food_left = dx > 0
            food_right = dx < 0
            food_front = dy > 0
            food_back = dy < 0
        elif self.direction == Vector2D.left:
            food_left = dy > 0
            food_right = dy < 0
            food_front = dx < 0
            food_back = dx > 0
        return [food_left, food_front, food_back, food_right]

    def convert_to_int(self, arr: list):
        summe = 0
        for i in range(len(arr)):
            num = int(arr[i])
            erg = (2 ** i) * num
            summe += erg
        return summe

                                        ### RENDERING ###
    #-------------------------------------------------------------------------------------------------------#
    def render_objects(self, screen: pygame.Surface):
        screen.fill((0,0,0))
        # show the grid
        self.show_grid(screen)
        # Render all parts
        self.render_parts(screen)
        # Render food
        self.render_food(screen)


    def render_parts(self, screen: pygame.Surface):
        """Render all snake parts"""
        for part_pos in self.part_list:
            screen_pos = grid_to_screen_pos(part_pos)
            part = pygame.Rect(screen_pos.x, screen_pos.y, snake_part_render_size, snake_part_render_size)
            t = self.part_list.index(part_pos) / (len(self.part_list) - 1)
            new_color = (part_color[0], int(part_color[1]* (1-t)) +105, part_color[2])
            pygame.draw.rect(screen, new_color, part)

    def render_food(self, screen: pygame.Surface):
        screen_pos = grid_to_screen_pos(self.food)
        food = pygame.Rect(screen_pos.x, screen_pos.y, food_render_size, food_render_size)
        pygame.draw.rect(screen, food_color, food)

    def show_grid(self, screen: pygame.Surface):
        for i in range(Grid.cell_count):
            for j in range(Grid.cell_count):
                cell = pygame.Rect((i * Grid.cell_size) + Grid.start_pos.x, (j * Grid.cell_size) + Grid.start_pos.y,
                                   Grid.cell_render_width, Grid.cell_render_width)
                pygame.draw.rect(screen, (200, 0, 0), cell)




