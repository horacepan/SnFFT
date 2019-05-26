from collections import namedtuple
import numpy as np

Batch = namedtuple('Batch', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.filled = 0

        self.states = np.empty([capacity], np.dtype('<U24'))
        self.actions = np.empty([capacity, 1], np.dtype('i1'))
        self.new_states = np.empty([capacity], np.dtype('<U24'))
        self.rewards = np.empty([capacity, 1], dtype=np.float32)
        self.dones = np.empty([capacity, 1], np.dtype(np.float32))

    def push(self, state, action, new_state, reward, done):
        self.states[self.position]      = state
        self.actions[self.position]     = action
        self.new_states[self.position]  = new_state
        self.rewards[self.position]     = reward
        self.dones[self.position]       = done

        self.position = (self.position + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.filled, batch_size)
        state = self.states[idx]
        new_state = self.new_states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        return Batch(state, action, new_state, reward, done)

    def __len__(self):
        return self.filled
