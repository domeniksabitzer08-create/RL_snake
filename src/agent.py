from random import randint
from time import sleep
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch import transpose

import snake_classes
from snake_classes import *

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

# Game Enviroment
env = SnakeManager(Vector2D(3,4),Vector2D.right,4, render=True)

score = 100




def train():
 pass

if __name__ == '__main__':
    train()
