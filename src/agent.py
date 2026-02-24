from random import randint
from time import sleep
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch import transpose

from cnn import  CNN
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

# Model
model = CNN(1,3,10)


def train():
    game_over = False
    action = [0, 1, 0]
    while not game_over:
        sleep(0.01)
        state, reward, game_over, score = env.step(action)
        # Convert state to Tensor
        state = torchvision.transforms.ToTensor()(state)
        # Divide numbers so they are in range 0-1
        state = state / 3
        # Convert the datatype
        state = state.type(torch.float32)
        # Add an extra dim (conv2d expects: (batch, channels, height, width))
        state = state.unsqueeze(0)
        # Pass data through the model
        y_logit = model(state)
        # from logit -> to prediction -> to action
        y_pred = torch.argmax(y_logit)
        action = [0,0,0]
        action[y_pred.item()] = 1

if __name__ == '__main__':
    train()
