import pdb
from collections import namedtuple
import numpy as np

Batch = namedtuple('Batch', ('state', 'action', 'next_state', 'reward', 'done', 'scramble_dist'))
Batch2 = namedtuple('Batch2', ('state', 'nbrs', 'grid', 'reward', 'done'))

class SimpleMemory(object):
    def __init__(self, capacity, dim_dict, dtype_dict):
        self.capacity = capacity
        self.position = 0
        self.filled = 0
        self.mem = {}
        for key, shape in dim_dict.items():
            self.mem[key] = np.empty((capacity,) + shape, dtype=dtype_dict.get(key, np.float32))

    def push(self, _dict):
        for k, v in _dict.items():
            self.mem[k][self.position] = v
        self.position = (self.position + 1) % self.capacity
        self.filled = min(self.capacity, self.filled + 1)

    def sample(self, batch_size):
        batch = {}
        idx = np.random.choice(self.filled, batch_size)
        for k, v in self.mem.items():
            batch[k] = self.mem[k][idx]
        return batch

    def __len__(self):
        return self.filled

    def prefill(self, dic):
        for k, v in dic.items():
            for i in range(self.capacity):
                self.mem[k][i] = v

class ReplayMemory(object):
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.position = 0
        self.filled = 0

        self.states = np.empty([capacity, state_dim], np.float32)
        self.actions = np.empty([capacity, 1], int)
        self.new_states = np.empty([capacity, state_dim], np.float32)
        self.scramble_dists = np.empty([capacity, 1], int)
        self.rewards = np.empty([capacity, 1], np.float32)
        self.dones = np.empty([capacity, 1], np.float32)

    def push(self, state, action, new_state, reward, done, scramble_dist):
        self.states[self.position]      = state
        self.actions[self.position]     = action
        self.new_states[self.position]  = new_state
        self.rewards[self.position]     = reward
        self.dones[self.position]       = done
        self.scramble_dists[self.position] = scramble_dist

        self.position = (self.position + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.filled, batch_size)
        state = self.states[idx]
        new_state = self.new_states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        scramble_dist = self.scramble_dists[idx]
        return Batch(state, action, new_state, reward, done, scramble_dist)

    def __len__(self):
        return self.filled

class ReplayMemory2(object):
    def __init__(self, capacity, state_dim, grid_shape):
        self.capacity = capacity
        self.position = 0
        self.filled = 0

        self.states = np.empty((capacity, state_dim), np.float32)
        self.nbrs = np.empty((capacity, 4, state_dim), np.float32)
        self.grids = np.empty((capacity,) + grid_shape, int)
        self.rewards = np.empty([capacity, 1], np.float32)
        self.dones = np.empty([capacity, 1], np.float32)

    def push(self, state, nbrs, grid, reward, done):
        self.states[self.position]      = state
        self.nbrs[self.position]        = nbrs
        self.grids[self.position]       = grid
        self.rewards[self.position]     = reward
        self.dones[self.position]       = done

        self.position = (self.position + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.filled, batch_size)
        state = self.states[idx]
        nbrs = self.nbrs[idx]
        grid = self.grids[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        return Batch2(state, nbrs, grid, reward, done)

    def __len__(self):
        return self.filled
