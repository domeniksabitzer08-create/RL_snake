import random
import time
from random import randint
from time import sleep
import numpy as np
from sympy.abc import epsilon

import snake_classes
from snake_classes import *


# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

# Game Enviroment
Env = SnakeManager(Vector2D(4,2),Vector2D.right,5, render=True)

score = 100
# action
N_actions = 3
# state
N_directions = 4
N_danger = 8 # 3 bits
N_food_dir = 16 # 4 bits
Q_table = np.zeros((N_directions*N_danger*N_food_dir, N_actions))
# hyperparameters
# train
Lr = 0.1
Gamma = 0.9
Epsilon = 1
Min_epsilon = 0.1
Epsilon_decay = 0.999
Num_episodes = 10000
Max_steps = 300
# test
global TestEnv
#TestEnv = SnakeManager(Vector2D(2,2),Vector2D.right,3, render=True)
global TestEpisodes
TestEpisodes = 100
# tracking
Train_score = 0
Life_span = 0

def choose_action(state, epsilon):
    if random.uniform(0, 1) <= epsilon:
        return Env.sample()
    else:
        return np.argmax(Q_table[state, :])

def train(lr, gamma, epsilon, epsilon_decay, num_episodes, max_steps):
    Life_span = 0
    Train_score = 0
    for i in range(num_episodes):
        is_game_over = False
        state = Env.reset()
        reward = 0
        for step in range(max_steps):
            # 0. calculate an action
            action = choose_action(state, epsilon)
            # 1. get the next state, reward, if the game is over and the score
            next_state, reward, is_game_over, score = Env.step(action)
            # 2. get the old value
            old_value = Q_table[state, action]
            # 3. get the new value
            next_value = np.max(Q_table[next_state])
            # 4. update the Q_table
            Q_table[state, action] = old_value + lr * (reward + gamma * next_value - old_value)
            # 5. set the new state as the current state
            state = next_state
            # 6. check if the game is over
            if is_game_over:
                Train_score += score
                Life_span += step +1
                break
            if i % 2000 == 0:
                sleep(0.1)
        # 7. decrease epsilon
        epsilon = max(Min_epsilon, epsilon * epsilon_decay)
    avg_life_span = Life_span / num_episodes
    avg_train_score = Train_score / num_episodes
    print(f"avg_life_span: {avg_life_span} | avg_train_score: {avg_train_score}")

def test():
    print("test")
    test_score = 0
    life_span = 0
    env = SnakeManager(Vector2D(3,2),Vector2D.right,5, render=True)
    for i in range(TestEpisodes):
        state = env.reset()
        score = 0
        reward = 0
        for step in range(200):
            action = choose_action(state,0)
            state, reward, is_game_over, score = env.step(action)
            sleep(0.2)
            if is_game_over:
                test_score += score
                life_span += step +1
                break
    avg_life_span = life_span / TestEpisodes
    avg_test_score = test_score / TestEpisodes
    print(f"avg_life_span TEST: {avg_life_span} | avg_score TEST: {avg_test_score}")

if __name__ == '__main__':
    train(Lr,Gamma, Epsilon, Epsilon_decay, Num_episodes, Max_steps)
    test()