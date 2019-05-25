import sys
sys.path.append('../')
from young_tableau import FerrersDiagram
import torch
import torch.nn as nn

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

def test():
    partitions = [(8, 1)]
    net = IrrepDQN(partitions)
    net.forward(torch.rand(100, 64))

if __name__ == '__main__':
    test()
