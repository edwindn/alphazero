import torch
import gymnasium as gym
import ale_py
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

gym.register_envs(ale_py)

#env = gym.make('MsPacman-v4', render_mode='human')
env = gym.make('Pong-v4', render_mode='human')
action_size = env.action_space.n # 9 for pacman, 6 for pong

def compress(state):
    im = np.array(state)
    im = im[1:176:2, ::2]
    im = im.mean(axis=2)
    im = np.expand_dims(im.reshape(1, 88, 80), axis=0)
    im = (im-128)/128 - 1
    return im

class DQN():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))

        self.gamma = 0.9
        self.sync_every = 1000
        self.burnin = 100
        self.epsilon = 0.8 # explore rate
        self.batch_size = 32

        self.net = Network(action_dim)
        self.loss_fn = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0005)

    def cache(self, state, action, reward, next_state, done):
        self.memory.add(TensorDict({
            "state": torch.tensor(state),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(next_state),
            "done": done
        }))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            Q_vals = self.net(state, model='online')
            return torch.argmax(Q_vals)
        
    def sync_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def train_step(self):
        batch = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = (batch.get(key) for key in ('state', 'action', 'reward', 'next_state', 'done'))
        th_action = torch.argmax(self.net(next_state.squeeze(1), model='online'), axis=1)
        target = reward + (1 - done.float()) * self.gamma * self.net(next_state.squeeze(1), model='target')[np.arange(self.batch_size), th_action]
        current = self.net(state.squeeze(1), model='online')[np.arange(self.batch_size), action]
        
        loss = self.loss_fn(target, current)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()
        
    def train(self): # copied function
        batch = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = (batch.get(key) for key in ('state', 'action', 'reward', 'next_state', 'done'))
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.net(next_state, model='target'))


class Network(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.target = self.cnn(1, action_dim)
        self.online = self.cnn(1, action_dim)

        self.target.load_state_dict(self.online.state_dict())

    def forward(self, x, model):
        x = torch.tensor(x, dtype=torch.float32)
        if model == 'online':
            return self.online(x)
        elif model == 'target':
            return self.target(x)

    def cnn(self, in_channels, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2688, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
if __name__ == '__main__':
    agent = DQN((88, 80, 1), action_size)

    num_episodes = 100
    num_timesteps = 10000

    for i in range(num_episodes):
        state = compress(env.reset()[0])

        for t in tqdm(range(num_timesteps)):
            env.render()
            if t % agent.sync_every == 0:
                agent.sync_target()
            
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = compress(next_state)
            agent.cache(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

            if t > agent.burnin:
                agent.train_step()

