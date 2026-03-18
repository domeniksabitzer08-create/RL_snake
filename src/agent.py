import random
import time
from collections import deque
from collections import namedtuple
import numpy as np
from sympy.abc import epsilon
from torch.distributed.argparse_util import env

from snake_classes import *
from tqdm.auto import tqdm
# importing PyTorch
import torch
from torch import nn, dtype
from torch.utils.data import DataLoader


class DQN(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=64):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
        )
    def forward(self, x):
        return self.layer_stack(x)

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

# Game Environment
Env = SnakeManager(Vector2D(4,2),Vector2D.right,3, render=False)


# action
N_actions = 3
N_states = 12
# hyperparameters
# train
Lr = 0.0001
Gamma = 1
Epsilon = 1
Min_epsilon = 0.01
Epsilon_decay = 0.999
Num_episodes = 10001
Max_steps = 1000
N_capacity = 10000
Batch_size = 32
Network_sync_rate = 500
# Test
Test_episodes = 100
# Epsilon-Greedy-Algorithm (take the best action or random one)
def choose_action(state, policy: torch.nn.Module):
    if np.random.random() <= Epsilon:
        return Env.sample()
    else:
        with torch.inference_mode():
            return torch.argmax(policy(state))


def test_model():
    # test data flow of model with dummy tensor
    model = DQN(10, 3)
    dummy_tensor = torch.rand(10)
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
# create instance of the model
policy_dqn = DQN(N_states, N_actions ).to(device)
target_dqn = DQN(N_states, N_actions ).to(device)
# load the state dict form the policy to the target dqn
target_dqn.load_state_dict(policy_dqn.state_dict())

# setup optimizer and loss function
optimizer = torch.optim.Adam(params=policy_dqn.parameters(), lr=Lr)
loss_fn = torch.nn.MSELoss()



def train(render:bool=False):
    Env = SnakeManager(Vector2D(5, 2), Vector2D.right, 5, render=render)
    print("Starting training...")
    # data variables
    train_loss = 0
    taken_steps = 0
    train_score = 0
    # assign epsilon
    global Epsilon
    # counts the number of steps taken
    step_count = 0
    # Initialize Experience Replay
    memory = ExperienceReplay(N_capacity, Batch_size)
    for episode in range(Num_episodes):
        episode_reward = 0
        is_done = False
        state = Env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)

        for step in range(Max_steps):
            # choose an action
            action = choose_action(state, policy_dqn)
            # make the action and receive values
            next_state, reward, is_done, score = Env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            # accumulate reward
            episode_reward += reward
            # add experience to the replay buffer
            memory.add_experience(state, action, reward, next_state, is_done)
            # increase counter
            step_count += 1
            # set the state to the new state
            state = next_state

            # update data
            taken_steps += 1


            # if enough experience has been collected, the nn can be optimized
            if memory.can_provide_sample():
                # get a batch of experiences
                batch = memory.sample_batch()

                # Create batches of experiences for faster computation and better optimization
                states = torch.stack([e.state for e in batch]).to(device)
                actions = torch.tensor([e.action for e in batch], dtype=torch.torch.long).to(device).unsqueeze(1)
                rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.stack([e.next_state for e in batch]).to(device)
                dones = torch.tensor([e.is_done for e in batch], dtype=torch.bool).unsqueeze(1).float().to(device)

                q_values = policy_dqn(states)
                with torch.no_grad():
                    q_values_next = target_dqn(next_states)

                # calculate target
                q_target = rewards + Gamma * q_values_next.max(dim=1, keepdim=True)[0] * (1 - dones)
                # get current q values
                current_q = q_values.gather(1, actions)

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
                    if episode % 1000 == 0:
                        train_score += score
                        # calculate avg data
                        avg_train_loss = train_loss / (episode +1)
                        avg_taken_steps = taken_steps / (episode +1)
                        avg_train_reward = episode_reward / (episode +1)
                        # reset data
                        train_loss = 0
                        taken_steps = 0
                        train_score = 0
                        try:
                            print(f"Episode: {episode} | avg. Loss: {avg_train_loss:.2f} | avg. taken steps {avg_taken_steps:.2f}| avg. reward  {avg_train_reward:.5f}")
                        except UnboundLocalError:
                            print(
                                f"Episode: {episode} | Loss: No Loss calculated yet | Reward: {episode_reward} | step {step} | is done: {is_done}")
                    break
        # Decay the epsilon
        Epsilon = max(Min_epsilon, Epsilon * Epsilon_decay)

def test():
    # Init new environment
    env = SnakeManager(Vector2D(5, 2), Vector2D.right, 5, render=True)
    test_loss = 0
    global Epsilon
    Epsilon = 0
    for episode in range(Test_episodes):
        episode_reward = 0
        is_done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)
        while not is_done:
            action = choose_action(state, policy_dqn)
            next_state, reward, is_done, score = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            state = next_state
            #print(f"state: {state}")
            time.sleep(0.1)



if __name__ == '__main__':
    train()
    #print(f"Epsilon: {Epsilon}")
    test()