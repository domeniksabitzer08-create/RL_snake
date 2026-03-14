import random
from time import sleep
import numpy as np
from torch.linalg import multi_dot

from snake_classes import *
from tqdm.auto import tqdm
# importing PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
        )
    def forward(self, x):
        return self.layer_stack(x)



# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

# Game Environment
Env = SnakeManager(Vector2D(4,2),Vector2D.right,5, render=True)

score = 100
# action
N_actions = 3
# hyperparameters
# train
Lr = 0.1
Gamma = 0.9
Epsilon = 1
Min_epsilon = 0.1
Epsilon_decay = 0.999
Num_episodes = 10000
Max_steps = 300

def test_model():
    # test data flow of model with dummy tensor
    model = Model(10, 3)
    dummy_tensor = torch.rand(10)
    # send tensor through model
    y_logit = model(dummy_tensor)
    y_pred = torch.argmax(torch.softmax(y_logit, dim=0))
    print(f"y_logit: {y_logit} | action: {y_pred}")

def check_state():
    env = SnakeManager(Vector2D(4, 2), Vector2D.right, 5, render=True)
    while True:
        state, reward, is_game_over, score = env.step([0,0,1])
        if is_game_over:
            break

if __name__ == '__main__':
    check_state()