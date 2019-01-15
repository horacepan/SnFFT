import pdb
from functools import reduce
from itertools import permutations
import time
from perm import Perm

SN_CACHE = {}
HITS = {'hits': 0}
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
        self.size = n
        self._map = self._filled_map(p_map, self.size)
        self.cycle_decomposition = self._cycle_decomposition() if cyc_decomp is None else cyc_decomp
        self.tup_rep = self.get_tup_rep() if tup_rep is None else tup_rep

        # add permutation to the cache
        if self.size in SN_CACHE:
            if (self.tup_rep not in SN_CACHE):
                SN_CACHE[self.size][self.tup_rep] = self
        elif self.size not in SN_CACHE:
            SN_CACHE[self.size] = {}
            SN_CACHE[self.size][self.tup_rep] = self

    def _filled_map(self, _map, size):
        for i in range(1, size+1):
            if i not in _map:
                _map[i] = i
        return _map

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
    def from_tup(tup):
        if len(tup) in SN_CACHE:
            if tup in SN_CACHE[len(tup)]:
                HITS['hits'] += 1
                return SN_CACHE[len(tup)][tup]
        else:
            SN_CACHE[len(tup)] = {}

        _dict = {idx+1: val for idx, val in enumerate(tup)}
        perm = Perm2(_dict, len(tup), tup)

        # store the thing
        SN_CACHE[len(tup)][perm.tup_rep] = perm
        return perm

    def __call__(self, x):
        return self._map.get(x, x)

    def __getitem__(self, x):
        return self._map.get(x, x)

    def __repr__(self):
        # mostly for debugging so dont care about recomputing this conversion
        return str(self.cycle_decomposition)

    def __mul__(self, other):
        g = self.tup_rep
        h = other.tup_rep
        new_tup = tuple(g[h[i] - 1] for i in range(self.size))
        return Perm2.from_tup(new_tup)

    # deprecated
    def mul2(self, other):
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
        rev_lst = [0] * self.size
        for idx, v in enumerate(self.tup_rep):
            # idx is 0 indexed, but v is 1 indexed?
            rev_lst[v-1] = idx + 1
        rev_tup = tuple(rev_lst) # surely theres a better way to do this?

        if rev_tup in SN_CACHE[self.size]:
            HITS['hits'] += 1
            return SN_CACHE[self.size][rev_tup]
        else:
            rev_map = {v: k for k, v in self._map.items()}
            return Perm2(rev_map, self.size)
        rev_map = {v: k for k, v in self._map.items()}
        return Perm2(rev_map, self.size)

 
def sn(n):
    if n in SN_CACHE:
        return list(SN_CACHE[n].values())

    perm_tups = permutations(range(1, n+1))
    perms = [Perm2.from_tup(t) for t in perm_tups]
    SN_CACHE[n] = {p.tup_rep: p for p in perms}
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
    x = Perm2({1:2, 2:1}, 4)
    y = Perm2({3:4, 4:3}, 4)
    z = x * y
    print(z)
    z2 = x.mul2(y)
    print(z2)
    print(z == z2)
