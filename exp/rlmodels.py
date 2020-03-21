import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, nin, nhid, nout, layers=2, to_tensor=None, std=0.1):
        super(MLP, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.fc_in = nn.Linear(nin, nhid)
        for i in range(1, layers):
            setattr(self, f'fc_h{i}', nn.Linear(nhid, nhid))
        self.fc_out = nn.Linear(nhid, nout)
        self.layers = [self.fc_in] + [getattr(self, f'fc_h{i}') for i in range(1, layers)] + [self.fc_out]
        self.nonlinearity = F.relu
        self.to_tensor = to_tensor
        self.reset_parameters(std)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.nonlinearity(x)
        return self.fc_out(x)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)

class ResidualBlock(nn.Module):
    def __init__(self, nin, nhid, nout, to_tensor, std=0.1):
        super(ResidualBlock, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.fc1 = nn.Linear(nin, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc_out = nn.Linear(nhid, nout)
        self.nonlin = F.relu
        self.to_tensor = to_tensor

    def forward(self, x):
        x_start = self.fc1(x)
        x = self.nonlin(self.fc1(x))
        x = self.nonlin(self.fc2(x))
        output = x + x_start
        return self.fc_out(output)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)

class LinRes(nn.Module):
    def __init__(self, nin, nhid, nout, to_tensor, std=0.1):
        super(LinRes, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.fc_in = nn.Linear(nin, nhid)
        self.res_block = ResidualBlock(nin, nhid, nout, to_tensor, std)
        self.nonlin = F.relu
        self.to_tensor = to_tensor

    def forward(self, x):
        x_start = self.fc1(x)
        x = self.nonlin(self.fc1(x))
        x = self.nonlin(self.fc2(x))
        output = x + x_start
        return self.fc_out(output)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)


class MLPResBlock(nn.Module):
    def __init__(self, nin, nhid, nout, to_tensor, std=0.1):
        super(MLPResBlock, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.mlp = MLP(nin, nhid, nhid, layers=1, to_tensor=to_tensor, std=std)
        self.res_block = ResidualBlock(nhid, nhid, nout, to_tensor, std)
        self.to_tensor = to_tensor

    def forward(self, x):
        x = self.mlp(x)
        return self.res_block(x)

class LinearPolicy(nn.Module):
    def __init__(self, nin, nout, to_tensor=None, std=0.1):
        super(LinearPolicy, self).__init__()
        self.nin = nin
        self.nout = 1
        self.net = nn.Linear(nin, nout)
        self.to_tensor = to_tensor

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)

class DQN(nn.Module):
    def __init__(self, nin, nhid, nout):
        pass

class DVN(nn.Module):
    def __init__(self, nin, nhid, nout):
        pass
