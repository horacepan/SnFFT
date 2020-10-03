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

class Linear(nn.Module):
    def __init__(self, nin, std=0.1, to_tensor=None):
        super(Linear, self).__init__()
        self.nin = nin
        self.net = nn.Linear(nin, 1)
        self.to_tensor = to_tensor

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)

class MLPBn(nn.Module):
    def __init__(self, nin, nhid, nout, layers=2, to_tensor=None, std=0.1):
        super(MLPBn, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.fc_in = nn.Linear(nin, nhid)
        for i in range(1, layers):
            setattr(self, f'fc_h{i}', nn.Linear(nhid, nhid))
        self.fc_out = nn.Linear(nhid, nout)
        self.layers = [self.fc_in] + [getattr(self, f'fc_h{i}') for i in range(1, layers)] + [self.fc_out]
        self.bns = nn.ModuleList([nn.BatchNorm1d(l.out_features) for l in self.layers[:-1]])
        self.nonlinearity = F.relu
        self.to_tensor = to_tensor
        self.reset_parameters(std)

    def forward(self, x):
        for layer, bn in zip(self.layers[:-1], self.bns):
            x = layer(x)
            x = bn(self.nonlinearity(x))
        return self.fc_out(x)

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)


class ResidualBlock(nn.Module):
    def __init__(self, nin, std=0.1):
        super(ResidualBlock, self).__init__()
        self.nin = nin
        self.fc1 = nn.Linear(nin, nin)
        self.fc2 = nn.Linear(nin, nin)
        self.nonlin = F.relu

    def forward(self, x):
        x_start = x
        x = self.nonlin(self.fc1(x))
        x = self.fc2(x)
        x = self.nonlin(x + x_start)
        return x

class MLPResModel(nn.Module):
    def __init__(self, nin, nh1, nh2, nout, nres, std=0.1, to_tensor=None):
        super(MLPResModel, self).__init__()
        self.nout = nout
        self.nres = nres
        self.to_tensor = to_tensor
        self.fc_1 = nn.Linear(nin, nh1)
        self.fc_2 = nn.Linear(nh1, nh2)
        self.fc_out = nn.Linear(nh2, nout)
        for i in range(nres):
            setattr(self, f'res_block_{i+1}', ResidualBlock(nh2))

        self.reset_parameters(std)

    def forward(self, x):
        xin = x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        for i in range(self.nres):
            res_layer = getattr(self, f'res_block_{i+1}')
            x = res_layer(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self, std=0.1):
        for p in self.parameters():
            p.data.normal_(std=std)

    def xinit(self):
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                pdb.set_trace()

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
