import pdb
from functools import reduce
from itertools import permutations
import time
from perm import Perm

SN_CACHE = {}

def conjugate(x, g):
    '''
    x: Perm2 object
    g: Perm2 object
    returns x conjugated by g
    '''
    return g.inv() * x * g
    
class ProdPerm:
    def __init__(self, *perms):
        '''
        perms: list of Perm2 objects
        '''
        self.perms = perms
        self.tup_rep = self.get_tup_rep()

    def __mul__(self, other):
        res = [a*b for a,b in zip(self.perms, other.perms)]
        return ProdPerm(*res)

    def get_tup_rep(self):
        return tuple(p.tup_rep for p in self.perms)

    def __repr__(self):
        return str(self.tup_rep)

    def inv(self):
        res = [p.inv() for p in self.perms]
        return ProdPerm(*res)

class Perm2:
    def __init__(self, p_map, n, tup_rep=None, cyc_decomp=None):
        self._map = p_map
        self.size = n
        self.cycle_decomposition = self._cycle_decomposition() if cyc_decomp is None else cyc_decomp
        self.tup_rep = self.get_tup_rep() if tup_rep is None else tup_rep

    def get_tup_rep(self):
        lst = []
        for i in range(1, len(self._map) + 1):
            lst.append(self._map.get(i, i))
        return tuple(lst)

    @staticmethod
    def eye(size):
        p_map = {i:i for i in range(1, size+1)}
        return Perm2(p_map, size)

    @staticmethod
    def from_lst(lst):
        _dict = {idx+1: val for idx, val in enumerate(lst)}
        return Perm2(_dict, len(lst))

    @staticmethod
    def from_tup(tup):
        _dict = {idx+1: val for idx, val in enumerate(tup)}
        return Perm2(_dict, len(tup), tup)

    def __call__(self, x):
        return self._map.get(x, x)

    def __getitem__(self, x):
        return self._map.get(x, x)

    def __repr__(self):
        # mostly for debugging so dont care about recomputing this conversion
        return str(self.cycle_decomposition)

    def __mul__(self, other):
        new_dict = {}
        n = max(self.size, other.size)
        for k in range(1, n+1):
            l = other._map.get(k, k)
            new_dict[k] = self._map.get(l, l)

        return Perm2(new_dict, n)

    def __len__(self):
        return self.size

    def __hash__(self):
        return hash(self.tup_rep)

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.tup_rep == other.tup_rep

    def _cycle_decomposition(self):
        cyc_decomp = []
        curr_cycle = []
        seen = set()
        for i in range(1, self.size+1):
            if i in seen:
                continue
            curr = i

            while True:
                if curr not in seen:
                    curr_cycle.append(curr)
                    seen.add(curr)
                    curr = self._map.get(curr, curr)
                else:
                    if len(curr_cycle) > 1:
                        cyc_decomp.append(curr_cycle)
                    curr_cycle = []
                    break
            seen.add(curr)

        return cyc_decomp

    def to_tup(self):
        return tuple(self._map[i] for i in range(1, self.size+1))

    def inv(self):
        rev_map = {v: k for k, v in self._map.items()}
        return Perm2(rev_map, self.size)
 
def sn(n):
    if n in SN_CACHE:
        return SN_CACHE[n]
    perm_lsts = permutations(range(1, n+1))
    perms = [Perm2.from_lst(lst) for lst in perm_lsts]
    SN_CACHE[n] = perms
    return perms

def test():
    for n in range(2,11):
        print('n = {}'.format(n))
        f = lambda x, y: x*y
        start = time.time()
        for i in range(10000):
            ps = [Perm([(i, i+1)]) for i in range(1, n+1)]
            p = reduce(f, ps)
        end = time.time()
        print('Time for orginal perm: {:.2f}'.format(end - start))

        start =time.time()
        for i in range(10000):
            ps = [Perm2({i:i+1, i+1:i}, n) for i in range(1, n+1)]
            p = reduce(f, ps)
        end = time.time()
        print('Time for perm with maps: {:.2f}'.format(end - start))

        print('=' * 10)
    #start = time.time()
    #for i in range(10000):
    #    ps = [Perm([(i, i+1)]) for i in range(1, n+1)]
    #    p = reduce(f, ps)
    #end = time.time()
    #print('Time for orginal perm 2nd time: {:.2f}'.format(end - start))

if __name__ == '__main__':
    x = Perm2({1:2, 2:1}, 2)
    y = Perm2({3:4, 4:3}, 4)
    print(x*y)
    print(y*x)
