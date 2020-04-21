import pdb
from wreath_puzzle import px_wreath_inv, px_wreath_mul, convert_cyc

GENERATORS = {
            'R': ((2, 6, 7, 3),(2, 6,  10, 7),(0,1,2,0,0,2,1,0),(0,1,0,0,0,1,1,0,0,1,0,0)),
            'L': ((1, 4, 8, 5),(4, 8,  12, 5),(2,0,0,1,1,0,0,2),(0,0,0,1,1,0,0,1,0,0,0,1)),
            'U': ((1, 2, 3, 4),(1, 2,  3,  4),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0)),
            'F': ((5, 8, 7, 6),(9, 12, 11, 10),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0)),
            'D': ((3, 7, 8, 4),(3, 7,  11, 8),(0,0,1,2,0,0,2,1),(0,0,0,0,0,0,0,0,0,0,0,0)),
            'B': ((1, 5, 6, 2),(1, 5,  9, 6),(1,2,0,0,2,1,0,0),(0,0,0,0,0,0,0,0,0,0,0,0))
}

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
        self.gens = convert_gens(GENERATORS)
        self.inv_gens = self._load_inv_gens()
        self.moves = [self.gens[f] for f in self._moves] + [self.inv_gens[f] for f in self._moves]

    def _load_inv_gens(self):
        inv_gens = {}
        for k, (t1, p1, t2, p2) in self.gens.items():
            t1i, p1i = px_wreath_inv(t1, p1, 3)
            t2i, p2i = px_wreath_inv(t2, p2, 2)
            inv_gens[k] = (t1i, p1i, t2i, p2i)
        return inv_gens

    def nbrs(self, state):
        _nbrs = []
        ct, cp, et, ep = state

        for (nct, ncp, net, nep) in self.moves:
            nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
            nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 3)
            _nbrs.append((nbr_ct, nbr_cp, nbr_et, nbr_ep))
        return _nbrs

    def step(self, state, action_idx):
        ct, cp, et, ep = state
        nct, ncp, net, nep = self.moves[action_idx]
        nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
        nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
        return (nbr_ct, nbr_cp, nbr_et, nbr_ep)

    def _step(self, state, move):
        '''
        state: 4 tuple of the (cube orientation tuple, cube permutation, edge orientation tuple, edge permutation)
        move: string
        '''
        ct, cp, et, ep = state
        nct, ncp, net, nep = self.gens[move]
        nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
        nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
        # order should be consistent with the ordering of the dictionaries
        return (nbr_ct, nbr_cp, nbr_et, nbr_ep)

    def _inv_step(self, state, move):
        ct, cp, et, ep = state
        nct, ncp, net, nep = self.inv_gens[move]
        nbr_ct, nbr_cp = px_wreath_mul(nct, ncp, ct, cp, 3)
        nbr_et, nbr_ep = px_wreath_mul(net, nep, et, ep, 2)
        return (nbr_ct, nbr_cp, nbr_et, nbr_ep)

def test():
    cube = Cube3()
    t1 = (0,) * 8
    p1 = tuple(range(1, 9))
    t2 = (0,) * 12
    p2 = tuple(range(1, 13))

    state = (t1, p1, t2, p2)
    for f in cube._moves:
        res = cube._step(state, f)
        ires = cube._inv_step(state, f)
        assert res == cube.gens[f]
        assert ires == cube.inv_gens[f]
    pdb.set_trace()

if __name__ == '__main__':
    test()
