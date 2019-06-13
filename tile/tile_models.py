import sys
sys.path.append('../')
import pdb
import math
from young_tableau import FerrersDiagram
import torch
import torch.nn as nn
import torch.nn.functional as F
from tile_env import grid_to_onehot

class IrrepDVN(nn.Module):
    def __init__(self, partitions):
        super(IrrepDVN, self).__init__()
        n_in = 0
        for p in partitions:
            f = FerrersDiagram(p)
            n_in += (len(f.tableaux) * len(f.tableaux))

        self.tile_size = sum(partitions[0])
        self.w = nn.Linear(n_in, 1)
        self.n_in = n_in
        self.n_out = 1
        self.init_weights()

    # Mostly for debugging
    def forward_grid(self, grid, env):
        irr = env.cat_irreps(grid)
        th_irrep = torch.from_numpy(irr).float().unsqueeze(0)
        return self.forward(th_irrep)

    def forward(self, x):
        '''
        Assumption is that x has already been raveled/concattenated so x is of dim: batch x n_in
        '''
        return self.w.forward(x)

    def init_weights(self):
        self.w.weight.data.normal_(0, 1. / math.sqrt(self.n_in + self.n_out))
        if self.tile_size == 2:
            self.w.bias[0] = -3.0
        elif self.tile_size == 3:
            self.w.bias[0] = -21.97

    def get_action(self, env, grid_state, e, all_nbrs=None, x=None, y=None):
        '''
        env: TileIrrepEnv
        state: not actually used! b/c we need to get the neighbors of the current state!
               Well, we _could_ have the state be the grid state!
        e: int
        '''
        if all_nbrs is None:
            all_nbrs, onehot_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

        invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
        vals = self.forward(torch.from_numpy(all_nbrs).float())
        # TODO: this is pretty hacky
        for m in invalid_moves:
            vals[m] = -float('inf')
        return torch.argmax(vals).item()

    def mem_dict(self, env):
        return {
            'grid_state': env.grid.shape,
            'next_grid_state': env.grid.shape,
            'irrep_state': (env.observation_space.shape[0],),
            'irrep_nbrs': (4, env.observation_space.shape[0]),
            'next_irrep_state': (env.observation_space.shape[0],),
            'action': (1,),
            'reward': (1,),
            'done': (1,),
            'dist': (1,),
            'scramble_dist': (1,),
        }

    def dtype_dict(self):
        return {
            'action': int,
            'scramble_dist': int,
        }

    def update(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch['reward'])
        dones = torch.from_numpy(batch['done'])
        states = torch.from_numpy(batch['irrep_state'])
        nbrs = torch.from_numpy(batch['irrep_nbrs'])
        grids = torch.from_numpy(batch['grid_state'])

        # Recall Q(s_t, a_t) = V(s_{t+1})
        pred_vals = self.forward(states)

        irrep_dim = states.size(-1)
        batch_all_nbrs = nbrs.view(-1, irrep_dim)
        all_next_vals = targ_net.forward(batch_all_nbrs)
        all_next_vals = all_next_vals.view(len(states), -1)
        best_vals = all_next_vals.max(dim=1)[0]
        targ_vals = rewards + discount * (1 - dones) * best_vals

        opt.zero_grad()
        loss = F.mse_loss(pred_vals, targ_vals.detach())
        loss.backward()
        opt.step()
        return loss.item()

class IrrepDQN(nn.Module):
    def __init__(self, partitions, nactions):
        super(IrrepDQN, self).__init__()
        n_in = 0
        for p in partitions:
            f = FerrersDiagram(p)
            n_in += (len(f.tableaux) * len(f.tableaux))

        self.tile_size = sum(partitions[0])
        self.w = nn.Linear(n_in, nactions)
        self.n_in = n_in
        self.n_out = nactions
        self.init_weights()

    def forward(self, x):
        '''
        Assumption is that x has already been raveled/concattenated so x is of dim: batch x n_in
        '''
        return self.w.forward(x)

    def forward_grid(self, grid, env):
        irr = env.cat_irreps(grid)
        th_irrep = torch.from_numpy(irr).float().unsqueeze(0)
        return self.forward(th_irrep)

    def init_weights(self):
        self.w.weight.data.normal_(0, 1. / math.sqrt(self.n_in + self.n_out))
        if self.tile_size == 2:
            self.w.bias[0] = -3.0
        elif self.tile_size == 3:
            self.w.bias[0] = -21.97

    def get_action_grid(self, env, grid, x, y):
        irrep = env.cat_irreps(grid)
        return self.get_action(env, irrep, x, y)

    def get_action(self, env, irrep_state, x, y):
        '''
        irrep_state: 1 x n_in tensor
        Returns int of the argmax
        '''
        irrep_state = torch.from_numpy(irrep_state).float().unsqueeze(0)
        invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
        vals = self.forward(irrep_state)
        for m in invalid_moves:
            vals[0, m] = -float('inf')
        return vals.argmax(dim=1).item()

    def update(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch['reward'])
        actions = torch.from_numpy(batch['action']).long()
        dones = torch.from_numpy(batch['done'])
        states = torch.from_numpy(batch['irrep_state'])
        next_states = torch.from_numpy(batch['next_irrep_state'])

        pred_vals = self.forward(states)
        vals = torch.gather(pred_vals, 1, actions)

        targ_max = targ_net.forward(next_states).max(dim=1)[0]
        targ_vals = (rewards + discount * (1 - dones) * targ_max.unsqueeze(-1)).detach()

        opt.zero_grad()
        loss = F.mse_loss(vals, targ_vals)
        loss.backward()
        opt.step()
        return loss.item()

    def mem_dict(self, env):
        '''
        env: TileIrrepEnv
        Assumption: The TileIrrepEnv's observation space will be the dimension of
        the irrep size.
        Returns a dictionary containing the sizes of the stuff the ReplayMemory has to store.
        '''
        return {
            'grid_state': env.grid.shape,
            'next_grid_state': env.grid.shape,
            'irrep_state': (env.observation_space.shape[0],),
            'irrep_nbrs': (4, env.observation_space.shape[0]),
            'next_irrep_state': (env.observation_space.shape[0],),
            'action': (1,),
            'reward': (1,),
            'done': (1,),
            'dist': (1,),
            'scramble_dist': (1,),
        }

    def dtype_dict(self):
        return {
            'action': int,
            'scramble_dist': int,
        }

class IrrepOnehotDVN(nn.Module):
    def __init__(self, onehot_shape, irrep_shape, n_hid, partitions):
        super(IrrepOnehotDVN, self).__init__()
        self.partitions = partitions
        self.tile_size = sum(partitions[0])
        self.n_in = onehot_shape[0]
        self.n_hid = n_hid
        self.irrep_size = irrep_shape[0]
        self.net = MLP(self.n_in, n_hid, irrep_shape[0])
        self.bias = nn.Parameter(torch.Tensor(1))
        self.init_weights()

    def forward(self, onehot, irrep):
        '''
        onehot: tensor of shape n x onehot_size
        irrep: tensor of shape n x irrep_size
        Returns: tensor of shape n x 1
        '''
        filt = self.net(onehot)
        vals = (filt * irrep).sum(dim=1) + self.bias
        return vals

    # Mostly for debugging
    def forward_grid(self, grid, env):
        irr = env.cat_irreps(grid)
        th_irrep = torch.from_numpy(irr).float().unsqueeze(0)
        th_onehot = torch.from_numpy(grid_to_onehot(grid)).float().unsqueeze(0)
        return self.forward(th_onehot, th_irrep)

    def get_action(self, env, grid_state, all_nbrs=None, x=None, y=None):
        # do we need the neighbors?
        irrep = env.cat_irreps(grid_state)
        if all_nbrs is None:
            all_nbrs, onehot_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

        invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
        vals = self.forward(torch.from_numpy(onehot_nbrs).float(), torch.from_numpy(all_nbrs).float())
        for m in invalid_moves:
            vals[m] = -float('inf')
        return torch.argmax(vals).item()

    def update(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch['reward'])
        dones = torch.from_numpy(batch['done'])
        irreps = torch.from_numpy(batch['irrep_state'])
        nbrs_irreps = torch.from_numpy(batch['irrep_nbrs'])
        nbrs_onehots = torch.from_numpy(batch['onehot_nbrs'])
        onehots = torch.from_numpy(batch['onehot_state']).float()

        # Recall Q(s_t, a_t) = V(s_{t+1})
        pred_vals = self.forward(onehots, irreps)
        all_next_vals = targ_net.forward(nbrs_onehots.view(-1, env.onehot_shape[0]),
                                         nbrs_irreps.view(-1, env.irrep_shape[0]))
        all_next_vals = all_next_vals.view(len(irreps), -1)
        best_vals = all_next_vals.max(dim=1)[0]
        targ_vals = rewards + discount * (1 - dones) * best_vals

        opt.zero_grad()
        loss = F.mse_loss(pred_vals, targ_vals.detach())
        loss.backward()
        opt.step()
        return loss.item()

    def init_weights(self):
        for p in self.parameters():
            p.data.normal_(0, 1. / math.sqrt(p.numel()))
        self.bias.data.zero_()

    def mem_dict(self, env):
        return {
            'grid_state': env.grid.shape,
            'onehot_state': env.onehot_shape,
            'next_grid_state': env.grid.shape,
            'irrep_state': (env.observation_space.shape[0],),
            'irrep_nbrs': (4, env.observation_space.shape[0]),
            'onehot_nbrs': (4, env.onehot_shape[0]),
            'next_irrep_state': (env.observation_space.shape[0],),
            'action': (1,),
            'reward': (1,),
            'done': (1,),
            'dist': (1,),
            'scramble_dist': (1,),
        }

    def dtype_dict(self):
        return {
            'action': int,
            'scramble_dist': int,
        }

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

    def get_action(self, env, grid_state, e, all_nbrs=None, x=None, y=None):
        '''
        env: TileIrrepEnv
        state: not actually used! b/c we need to get the neighbors of the current state!
               Well, we _could_ have the state be the grid state!
        e: int
        '''
        if all_nbrs is None:
            all_nbrs, onehot_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

        invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
        vals = self.forward(torch.from_numpy(all_nbrs).float())
        # TODO: this is pretty hacky
        for m in invalid_moves:
            vals[m] = -float('inf')
        return torch.argmax(vals).item()

    def forward_grid(self, grid, env):
        irr = env.cat_irreps(grid)
        th_irrep = torch.from_numpy(irr).float().unsqueeze(0)
        return self.forward(th_irrep)

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

    def update_simple(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch['reward'])
        actions = torch.from_numpy(batch['action']).long()
        dones = torch.from_numpy(batch['done'])
        states = torch.from_numpy(batch['onehot_state'])
        next_states = torch.from_numpy(batch['next_onehot_state'])

        pred_vals = self.forward(states)
        vals = torch.gather(pred_vals, 1, actions)

        targ_max = targ_net.forward(next_states).max(dim=1)[0]
        targ_vals = (rewards + discount * (1 - dones) * targ_max.unsqueeze(-1)).detach()

        opt.zero_grad()
        loss = F.mse_loss(vals, targ_vals)
        loss.backward()
        opt.step()
        return loss.item()


    def update(self, targ_net, env, batch, opt, discount, ep):
        '''
        targ_net: TileBaselineQ
        env: TileEnv
        batch: dictionary
        opt: torch optimizer
        discount: float
        ep: int, episode number

        Computes the loss and takes a gradient step.
        '''
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

    def get_action(self, env, grid_state, e, all_nbrs=None, x=None, y=None):
        # get neighbors
        if all_nbrs is None:
            all_nbrs, onehot_nbrs = env.all_nbrs(grid_state, x, y) # these are irreps

        invalid_moves = [m for m in env.MOVES if m not in env.valid_moves(x, y)]
        vals = self.forward(torch.from_numpy(all_nbrs).float())
        # TODO: this is pretty hacky
        for m in invalid_moves:
            vals[m] = -float('inf')
        return torch.argmax(vals).item()

    def update(self, targ_net, env, batch, opt, discount, ep):
        rewards = torch.from_numpy(batch['reward'])
        dones = torch.from_numpy(batch['done'])
        states = torch.from_numpy(batch['irrep_state'])
        next_states = torch.from_numpy(batch['next_irrep_state'])
        dists = torch.from_numpy(batch['scramble_dist']).float()
        pred_vals = self.forward(next_states)
        targ_vals = (rewards + discount * (1 - dones) * targ_net.forward(next_states))

        opt.zero_grad()
        #errors = (1 / (dists + 1.)) * (pred_vals - targ_vals.detach()).pow(2)
        #errors = (pred_vals - targ_vals.detach()).pow(2)
        #loss = errors.sum() / len(targ_vals)
        loss = F.mse_loss(pred_vals, targ_vals.detach())
        loss.backward()
        opt.step()
        return loss.item()



def test():
    partitions = [(8, 1)]
    net = IrrepDQN(partitions, 4)
    net.forward(torch.rand(100, 64))

if __name__ == '__main__':
    test()
