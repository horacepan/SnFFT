import pdb
import random
from wreath_puzzle import px_wreath_inv, px_wreath_mul, convert_cyc
from utility import wreath_onehot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUBE3_GENS = [
    ((0, 0, 0, 0, 0, 0, 0, 0), (2, 3, 4, 1, 5, 6, 7, 8), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (2, 3, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12)),
    ((0, 0, 1, 2, 0, 0, 2, 1), (1, 2, 7, 3, 5, 6, 8, 4), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 7, 4, 5, 6, 11, 3, 9, 10, 8, 12)),
    ((2, 0, 0, 1, 1, 0, 0, 2), (4, 2, 3, 8, 1, 6, 7, 5), (0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1), (1, 2, 3, 8, 4, 6, 7, 12, 9, 10, 11, 5)),
    ((0, 1, 2, 0, 0, 2, 1, 0), (1, 6, 2, 4, 5, 7, 3, 8), (0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), (1, 6, 3, 4, 5, 10, 2, 8, 9, 7, 11, 12)),
    ((0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 8, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 11)),
    ((1, 2, 0, 0, 2, 1, 0, 0), (5, 1, 3, 4, 6, 2, 7, 8), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (5, 2, 3, 4, 9, 1, 7, 8, 6, 10, 11, 12)),
    ((0, 0, 0, 0, 0, 0, 0, 0), (4, 1, 2, 3, 5, 6, 7, 8), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)),
    ((0, 0, 1, 2, 0, 0, 2, 1), (1, 2, 4, 8, 5, 6, 3, 7), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 8, 4, 5, 6, 3, 11, 9, 10, 7, 12)),
    ((2, 0, 0, 1, 1, 0, 0, 2), (5, 2, 3, 1, 8, 6, 7, 4), (0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1), (1, 2, 3, 5, 12, 6, 7, 4, 9, 10, 11, 8)),
    ((0, 1, 2, 0, 0, 2, 1, 0), (1, 3, 7, 4, 5, 2, 6, 8), (0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0), (1, 7, 3, 4, 5, 2, 10, 8, 9, 6, 11, 12)),
    ((0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 6, 7, 8, 5), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 9)),
    ((1, 2, 0, 0, 2, 1, 0, 0), (2, 6, 3, 4, 1, 5, 7, 8), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (6, 2, 3, 4, 1, 9, 7, 8, 5, 10, 11, 12)),
]

CUBE3_START = (0,) * 8, tuple(range(1, 9)), (0,) * 12, tuple(range(1, 13))

def convert_gens(generators):
    gens = {}
    for k, (p1, p2, t1, t2) in generators.items():
        _p1 = convert_cyc(p1, 8)
        _p2 = convert_cyc(p2, 12)
        gens[k] = (t1, _p1, t2, _p2)
    return gens

class Cube3(object):
    def __init__(self):
        self._moves = ['U', 'D', 'L', 'R', 'F', 'B']
        self.moves = CUBE3_GENS

    def start_state(self):
        return CUBE3_START

    def nbrs(self, state):
        _nbrs = []
        ct, cp, et, ep = state

        for (nct, ncp, net, nep) in self.moves:
            nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
            nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
            _nbrs.append((nbr_ct, nbr_cp, nbr_et, nbr_ep))
        return _nbrs

    def to_tensor(self, states):
        corner_tups = [(s[0], s[1]) for s in states]
        edge_tups = [(s[2], s[3]) for s in states]
        t1 = wreath_onehot(corner_tups, 3, cache=False)
        t2 = wreath_onehot(edge_tups, 2, cache=False)
        return torch.cat([t1, t2], dim=1).to(device)

    def step_idx(self, state, idx):
        '''
        state: 4 tuple of the (cube orientation tuple, cube permutation, edge orientation tuple, edge permutation)
        move: string
        '''
        move = self.moves[idx]
        return self.step(state, move)

    def step(self, state, move):
        ct, cp, et, ep = state
        nct, ncp, net, nep = move
        nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
        nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
        return (nbr_ct, nbr_cp, nbr_et, nbr_ep)

    def _inv_step(self, state, move):
        ct, cp, et, ep = state
        nct, ncp, net, nep = self.inv_gens[move]
        nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
        nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
        return (nbr_ct, nbr_cp, nbr_et, nbr_ep)

    def is_done(self, state):
        return state == CUBE3_START

    def random_walk(self, length):
        state = self.start_state()
        states = [state]

        for _ in range(length):
            move = random.choice(self.moves)
            state =  self.step(state, move)
            states.append(state)
        return states

    def random_move(self):
        return random.choice(self.moves)

    def random_state(self, length):
        state = self.start_state()
        for _ in range(length):
            move = random.choice(self.moves)
            state = self.step(state, move)
        return state

def test():
    cube = Cube3()
    state = CUBE3_START
    nbrs = cube.nbrs(state)
    tnbrs = cube.to_tensor(nbrs)
    print(tnbrs.shape)
    pdb.set_trace()

if __name__ == '__main__':
    test()
