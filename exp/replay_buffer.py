import numpy as np
import torch

class ReplayBufferMini:
    def __init__(self, capacity):
        self.state_objs = [None] * capacity
        self.next_state_objs = [None] * capacity
        self.rewards = torch.zeros(capacity, 1)
        self.actions = torch.zeros(capacity, 1)
        self.dones = torch.zeros(capacity, 1)
        self.capacity = capacity
        self.filled = 0
        self._idx = 0

    def push(self, state, action, next_state, reward, done):
        self.state_objs[self._idx] = state
        self.next_state_objs[self._idx] = next_state
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.dones[self._idx] = done
        self._idx = (self._idx + 1) % self.capacity
        self.filled = min(self.capacity, self.filled + 1)

    def sample(self, batch_size, device):
        '''
        Returns a tuple of the state, next state, reward, dones
        '''
        size = min(self.filled, batch_size)
        idxs = np.random.choice(self.filled, size=size)
        state_objs = [self.state_objs[i] for i in idxs]
        next_objs = [self.next_state_objs[i] for i in idxs]

        ba = self.actions[idxs].to(device)
        br = self.rewards[idxs].to(device)
        bd = self.dones[idxs].to(device)
        return state_objs, ba, next_objs, br, bd
