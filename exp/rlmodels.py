import pdb
import sys
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RlPolicy(nn.Module):
    def __init__(self, nin, nout, to_tensor):
        super(RlPolicy, self).__init__()
        self.nin = nin
        self.nin = nin
        self.net = nn.Linear(nin, nout)
        self.to_tensor = to_tensor

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        self.net.weight.data.normal(std=0.1)

    def forward_tup(self, tup):
        tens = self.to_tensor(tup).to(device)
        return self.forward(tens)

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = torch.cat([self.to_tensor(tup) for tup in tup_nbrs], dim=0).to(device)
        return self.forward(tens_nbrs).argmax()

class MLP(nn.Module):
    def __init__(self, nin, nhid, nout, to_tensor):
        super(MLP, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.net = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout)
        )
        self.to_tensor = to_tensor

    def forward(self, x):
        return self.net(x)

    def forward_tup(self, tup):
        tens = self.to_tensor(tup).to(device)
        return self.forward(tens)

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.forward(tens_nbrs).argmax()

class MLPMini(nn.Module):
    def __init__(self, nin, nhid, nout, to_tensor):
        super(MLPMini, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.net = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout),
        )
        self.to_tensor = to_tensor

    def forward(self, x):
        return self.net(x)

    def forward_tup(self, tup):
        tens = self.to_tensor(tup).to(device)
        return self.forward(tens)

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.forward(tens_nbrs).argmax()
