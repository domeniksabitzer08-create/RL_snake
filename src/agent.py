import random
import time
from random import randint
from time import sleep
import numpy as np


import snake_classes
from snake_classes import *


# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

# Game Enviroment
Env = SnakeManager(Vector2D(2,2),Vector2D.right,3, render=False)

score = 100
# action
N_actions = 3
# state
N_directions = 4
N_danger = 8 # 3 bits
N_food_dir = 16 # 4 bits
Q_table = np.zeros((N_directions*N_danger*N_food_dir, N_actions))
# hyperparameters
Lr = 0.1
Gamma = 0.9
Epsilon = 1
Min_epsilon = 0.05
Epsilon_decay = 0.995
Num_episodes = 100000
Max_steps = 100

# tracking
Train_score = 0
Life_span = 0

def choose_state(state):
    if random.uniform(0, 1) <= Epsilon:
        return Env.sample()
    else:
        return np.argmax(Q_table[state, :])

def train():
    Life_span = 0
    Train_score = 0
    for i in range(Num_episodes):
        is_game_over = False
        state = Env.reset()
        reward = 0
        for step in range(Max_steps):
            # 0. calculate an action
            action = choose_state(state)
            # 1. get the next state, reward, if the game is over and the score
            next_state, reward, is_game_over, score = Env.step(action)
            # 2. get the old value
            old_value = Q_table[state, action]
            # 3. get the new value
            next_value = np.max(Q_table[next_state])
            # 4. update the Q_table
            Q_table[state, action] = old_value + Lr * (reward + Gamma * next_value) - old_value
            # 5. set the new state as the current state
            state = next_state
            # 6. check if the game is over

            if is_game_over:
                print(f"episode: {i}, score: {score}")
                Train_score += score
                Life_span += step
                break
    avg_life_span = Life_span / Num_episodes
    avg_train_score = Train_score / Num_episodes
    print(f"avg_life_span: {avg_life_span} | avg_train_score: {avg_train_score}")
if __name__ == '__main__':
    train()
