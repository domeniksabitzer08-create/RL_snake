from random import randint
from time import sleep

from sympy.codegen.cnodes import static

import snake_classes
from snake_classes import *

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

env = SnakeManager(Vector2D(3,4),Vector2D.right,4, render=False)
game_over = False
action = [0,1,0]
score = 100

# DEBUG FUNCTIONS
def model(state):
    return [randint(0,1),randint(0,1),randint(0,1)]

while not game_over:
    sleep(0.01)
    state, reward, game_over, score = env.step(action)
    action = model(state)




