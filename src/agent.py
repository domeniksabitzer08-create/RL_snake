import copy
import random
import time
from collections import deque
from collections import namedtuple

import numpy as np
import pygame.time
import os

from snake_classes import *
from tqdm.auto import tqdm
# importing PyTorch
import torch
from torch import nn
# importing tensorboard
from torch.utils.tensorboard import SummaryWriter

### --------------------- SETUP --------------------- ###
Lr = 0.0001
Num_episodes = 300
Max_steps = 100
use_existing_model = False
used_model_name = "DQCNN_256_V_17"
Training = not use_existing_model
Experiment_name = "DQCNN_small_Grid_debug" + str(time.time())


### ENVIRONMENT ###
global Train_Env
Train_Env = SnakeManager(Vector2D(3,3),Vector2D.right,5, render=False)
global Test_Env
Test_Env = SnakeManager(Vector2D(3,3),Vector2D.right,5, render=True)



class DQN(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.out_features),
        )
    def forward(self, x):
        return self.layer_stack(x)

class DQCNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=out_features),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity) # create a double ended queue
        # Save information about the training in a named tuple
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "is_done"] )

    def add_experience(self, state, action, reward, next_state, is_done):
        # Create a new exp and save it into memory
        experience = self.Experience(state, action, reward, next_state, is_done)
        self.memory.append(experience)

    def sample_batch(self):
        # return random samples at the length of batch size
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        """returns True if enough samples are available for batch sampling"""
        return len(self.memory) >= self.batch_size

    def __len__(self):
        return len(self.memory)

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)

### SAVING AND LOADING MODEL ###
def save_model(model: torch.nn.Module):
    base_path = r"C:\Users\domen_s6zwlxv\PycharmProjects\RL_snake\models"
    name = f"{model.__class__.__name__}_{model.hidden_units}_V_{len(os.listdir(base_path))}"
    torch.save(model, fr"{base_path}\{name}.pth")
    print(f"saved model under name: {name}")
    return fr"{base_path}\{name}.pth"

def load_model(model_name: str):
    base_path = r"C:\Users\domen_s6zwlxv\PycharmProjects\RL_snake\models"
    model_name = fr"{base_path}\{model_name}.pth"
    try:
        model = torch.load(model_name, weights_only=False)
        print(f"model {model_name}.pth was loaded| type: {type(model)}")
        return model
    except FileNotFoundError:
        print(f"{model_name} was not found!")
        raise FileNotFoundError

# action
N_actions = 4
N_states = 3
# hyperparameters
# train
Gamma = 0.99
Epsilon = 1
Min_epsilon = 0.02
Epsilon_decay = 0.999
N_capacity = 100000
Batch_size = 32
Network_sync_rate = 1000
# Tensorboard
if Training:
    BASE_DIR = r"C:\Users\domen_s6zwlxv\PycharmProjects\RL_snake"
    runs_path = BASE_DIR + r"\runs"
    exp_path = runs_path + fr"\{Experiment_name}"
    os.mkdir(exp_path)
    Writer = SummaryWriter(str(exp_path))
# Test
Test_episodes = 100

# Epsilon-Greedy-Algorithm (take the best action or random one)
def choose_action(state, policy: torch.nn.Module, env: SnakeManager):
    if np.random.random() <= Epsilon:
        return env.sample()
    else:
        with torch.inference_mode():
            y_logit = policy(state)
            y_pred = torch.argmax(y_logit)
            return torch.argmax(policy(state))


def test_model():
    # test data flow of model with dummy tensor
    print("test model")
    model = DQCNN(3, 3)
    dummy_tensor = torch.rand((1,3,11,11))
    # send tensor through model
    y_logit = model(dummy_tensor)
    y_pred = torch.argmax(torch.softmax(y_logit, dim=0))
    y_max = y_logit.max()

    print(f"y_logit: {y_logit} | action: {y_pred} | y_max: {y_max}")

def check_state():
    env = SnakeManager(Vector2D(4, 2), Vector2D.right, 5, render=True)
    while True:
        state, reward, is_game_over, score = env.step([0,0,1])
        print(f"N states: {len(state)}")
        if is_game_over:
            break

### SETUP ###
# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"using device: {device} ")
# create instance of the model

# ---- Create Model or use existing ---- #
if use_existing_model:
    policy_dqn = load_model(used_model_name).to(device)
else:
    policy_dqn = DQCNN(N_states, N_actions).to(device)

target_dqn = copy.deepcopy(policy_dqn)
# load the state dict form the policy to the target dqn
target_dqn.load_state_dict(policy_dqn.state_dict())

# setup optimizer and loss function
optimizer = torch.optim.Adam(params=policy_dqn.parameters(), lr=Lr)
loss_fn = torch.nn.MSELoss()



def train(render:bool=False):
    global Train_Env
    Env = Train_Env
    print("Starting training...")
    # data tracking
    train_data_tracking = {
        "episode": [],
        "reward": [],
        "steps": [],
        "score": [],
        "epsilon": []
    }
    # data variables
    train_loss = 0
    avg_taken_steps = 0
    train_score = 0
    # assign epsilon
    global Epsilon
    # counts the number of steps taken
    step_count = 0
    # Initialize Experience Replay
    memory = ExperienceReplay(N_capacity, Batch_size)
    for episode in tqdm(range(Num_episodes)):
        episode_score = 0
        taken_steps = 0
        episode_reward = 0
        is_done = False
        state = Env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)
        if episode == 0:
            print(f"shape of one state: {state.shape}")
            print(f"1. channel: \n {state[0]}")
            print(f"2. channel: \n {state[1]}")
            print(f"3. channel: \n {state[2]}")
        for step in range(Max_steps):
            # choose an action
            state_for_model = state.unsqueeze(dim=0)
            action = choose_action(state_for_model, policy_dqn, Env)
            # make the action and receive values
            next_state, reward, is_done, score = Env.step(action)
            # if the max step range is reached, give a Game over Reward
            if step == Max_steps - 1:
                reward = -1
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            # accumulate reward
            episode_reward += reward
            # add experience to the replay buffer
            memory.add_experience(state, action, reward, next_state, is_done)
            # increase counter
            step_count += 1
            # set the state to the new state
            state = next_state

            # update steps and score
            taken_steps += 1
            avg_taken_steps += 1
            episode_score += reward


            # if enough experience has been collected, the nn can be optimized
            if memory.can_provide_sample():
                # get a batch of experiences
                batch = memory.sample_batch()
                #print("start real training")
                # Create batches of experiences for faster computation and better optimization
                states = torch.stack([e.state for e in batch]).to(device)
                actions = torch.tensor([e.action for e in batch], dtype=torch.torch.int64).to(device)
                rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.stack([e.next_state for e in batch]).to(device)
                dones = torch.tensor([e.is_done for e in batch], dtype=torch.bool).unsqueeze(1).float().to(device)
                #print(f"shape of batch of states: {states.shape}")
                q_values = policy_dqn(states)
                with torch.no_grad():
                    q_values_next = target_dqn(next_states)

                # calculate target
                q_target = rewards + Gamma * q_values_next.max(dim=1, keepdim=True)[0] * (1 - dones)
                # get current q values
                #print(f"actions shape: {actions.shape}")
                #print(f"q_values shape: {q_values.shape} ")
                current_q = q_values.gather(1, actions.unsqueeze(1))

                # calculate the loss
                loss = loss_fn(current_q, q_target)
                train_loss += loss.item()
                # optimizer zero grad
                optimizer.zero_grad()
                # loss backward
                loss.backward()
                # optimizer step
                optimizer.step()


                # Sync the target with the policy network
                if step_count % Network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())


            if is_done or step >= Max_steps:
                # append data
                train_data_tracking["episode"].append(episode)
                train_data_tracking["reward"].append(episode_reward)
                train_data_tracking["steps"].append(taken_steps)
                train_data_tracking["score"].append(episode_score)
                train_data_tracking["epsilon"].append(Epsilon)
                # add to data to tensorboard
                Writer.add_scalar("reward", episode_reward, episode)
                Writer.add_scalar("score", episode_score, episode)
                Writer.add_scalar("epsilon", Epsilon, episode)
                Writer.add_scalar("steps", taken_steps, episode)
                if episode % 1000 == 0 and episode != 0:
                    train_score += score
                    # calculate avg data
                    avg_train_loss = train_loss / (episode +1)
                    avg_taken_steps = avg_taken_steps / (episode +1)
                    avg_train_reward = episode_reward / (episode +1)

                    try:
                        print(f"Episode: {episode} | avg. Loss: {avg_train_loss:.2f} | avg. taken steps {avg_taken_steps:.2f}| avg. reward  {avg_train_reward:.5f}")
                    except UnboundLocalError:
                        print(
                            f"Episode: {episode} | Loss: No Loss calculated yet | Reward: {episode_reward} | step {step} | is done: {is_done}")
                    # reset data after it has been printed
                    train_loss = 0
                    avg_taken_steps = 0
                    train_score = 0

                break
        # Decay the epsilon
        Epsilon = max(Min_epsilon, Epsilon * Epsilon_decay)
    # Save model
    print("training ended")
    print(train_data_tracking)
    model_path = save_model(policy_dqn)

def test():
    # Init new environment
    clock = pygame.time.Clock()
    global Test_Env
    env = Test_Env
    test_loss = 0
    global Epsilon
    Epsilon = 0
    for episode in range(Test_episodes):
        episode_reward = 0
        is_done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)
        while not is_done:
            action = choose_action(state.unsqueeze(dim=0), policy_dqn, env)
            next_state, reward, is_done, score = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            state = next_state
            # Pygame rendering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            clock.tick(10)



if __name__ == '__main__':
    os.chdir(r"C:\Users\domen_s6zwlxv\PycharmProjects\RL_snake")
    if Training:
        train()

    test()