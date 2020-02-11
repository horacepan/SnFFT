import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, nin, nhid, nout, layers=2, to_tensor):
        super(MLP, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.fc_in = nn.Linear(nin, nhid)
        self.fc_hid = [nn.Linear(nhid, nhid) for _ in range(layers-1)]
        self.fc_out = nn.Linear(nhid, nout)
        self.layers = [self.fc_in] + self.fc_hid + [self.fc_out]
        self.nonlinearity = F.relu
        self.to_tensor = to_tensor

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.layer(x)
            x = self.nonlinearity(x)
        return self.fc_out(x)

    def forward_tup(self, tup):
        tens = self.to_tensor(tup).to(device)
        return self.forward(tens)

    def opt_move_tup(self, tup_nbrs):
        tens_nbrs = self.to_tensor(tup_nbrs).to(device)
        return self.forward(tens_nbrs).argmax()

    def eval_opt_nbr(self, nbr_tups, nnbrs):
        nbr_eval = self.forward_tup(nbr_tups).reshape(-1, nnbrs)
        max_nbr_vals = nbr_eval.max(dim=1, keepdim=True)[0]
        return max_nbr_vals
