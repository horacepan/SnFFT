import sys
sys.path.append('../')
from young_tableau import FerrersDiagram
import torch
import torch.nn as nn
import torch.nn.functional as F

class IrrepDQN(nn.Module):
    def __init__(self, partitions):
        super(IrrepDQN, self).__init__()

        n_in = 0
        for p in partitions:
            f = FerrersDiagram(p)
            n_in += (len(f.tableaux) * len(f.tableaux))

        self.w = nn.Linear(n_in, 1)
        self.n_in = n_in
        self.init_weights()

    def forward(self, x):
        '''
        Assumption is that x has already been raveled/concattenated so x is of dim: batch x n_in
        '''
        return self.w.forward(x)

    def init_weights(self):
        pass

class IrrepDQNMLP(nn.Module):
    def __init__(self, partition, n_hid, n_out):
        super(IrrepDQNMLP, self).__init__()
        ferrer = FerrersDiagram(partition)
        size = len(ferrer.tableaux) * len(ferrer.tableaux)
        self.net = MLP(size, n_hid, n_out)

    def forward(self, x):
        '''
        x: an irrep
        '''
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out)
        )

    def forward(self, x):
        return self.net.forward(x)

class TileBaselineQ(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(TileBaselineQ, self).__init__()
        self.net = MLP(n_in, n_hid, n_out)

    def get_action(self, state):
        '''
        state: 1 x n_in tensor
        Returns int of the argmax
        '''
        state = torch.from_numpy(state).float().unsqueeze(0)
        vals = self.forward(state)
        return vals.argmax(dim=1).item()

    def update(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch.reward)
        actions = torch.from_numpy(batch.action).long()
        dones = torch.from_numpy(batch.done)
        states = torch.from_numpy(batch.state)
        next_states = torch.from_numpy(batch.next_state)

        pred_vals = self.forward(states)
        vals = torch.gather(pred_vals, 1, actions)

        targ_max = targ_net.forward(next_states).max(dim=1)[0]
        targ_vals = (rewards + discount * (1 - dones) * targ_max.unsqueeze(-1)).detach()

        opt.zero_grad()
        loss = F.mse_loss(vals, targ_vals)
        loss.backward()
        opt.step()
        return loss.item()

    def forward(self, x):
        return self.net.forward(x)

class TileBaselineV(nn.Module):
    def __init__(self, n_in, n_hid):
        super(TileBaselineV, self).__init__()
        self.net = MLP(n_in, n_hid, 1)

    def get_action(self, states):
        pass

    def update(self):
        pass

def test():
    partitions = [(8, 1)]
    net = IrrepDQN(partitions)
    net.forward(torch.rand(100, 64))

if __name__ == '__main__':
    test()
