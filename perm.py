import sys
from utils import canonicalize, lcm
from itertools import permutations
import numpy as np
import pdb
from collections import deque

# Currently this just stores the entirety of S_n as values of the dict
# alternative is to store n -> dict mapping {partition -> Permutation object}
SN_CACHE = {}
class Perm:
    '''
    Assume the cycle decomposition given is in canonical form(no repeated things).
    Not necessarily in simplest form.
    '''
    def __init__(self, cyc_decomp):
        self._str = None
        try:
            self._max = max(map(lambda z: max(z), cyc_decomp)) # might not be accurate
        except:
            pdb.set_trace()
        self._map = self._init_map(cyc_decomp)
        self.decomp = self.canonical_cycle_decomp(cyc_decomp)
        self.tup_rep = self.get_tup_rep()

    def get_tup_rep(self):
        lst = []
        for i in range(1, len(self._map) + 1):
            lst.append(self._map.get(i, i))
        return tuple(lst)

    # TODO: this is really janky
    @staticmethod
    def from_dict(_dict):
        cyc_decomp = []
        curr_cyc = []
        curr = None
        seen = set()
        to_visit = deque([1])
        all_nodes = set(range(1, len(_dict) + 1))

        while len(seen) < len(_dict) or len(curr_cyc) > 0:
            if to_visit:
                curr = to_visit.popleft()
            else:
                curr = min(all_nodes)

            if curr in seen:
                cyc_decomp.append(tuple(curr_cyc))
                curr_cyc = []
            else:
                curr_cyc.append(curr)
                if curr in _dict:
                    to_visit.append(_dict[curr])

            seen.add(curr)
            if curr in all_nodes:
                all_nodes.remove(curr)

        return Perm(cyc_decomp)

    @staticmethod
    def from_lst(lst):
        _dict = {idx+1: val for idx, val in enumerate(lst)}
        return Perm.from_dict(_dict)

    @property
    def cycle_decomposition(self):
        return self.decomp

    def _init_map(self, cyc_decomp):
        cycle_dict = {}
        # this should be evaluated right to left
        elements = range(1, self._max + 1)

        for e in elements:
            # eval the permutation of e at each cycle
            e_mapped = e
            for cyc in cyc_decomp[::-1]:
                if e_mapped in cyc:
                    idx = cyc.index(e_mapped)
                    e_mapped = cyc[(idx + 1) % len(cyc)]
                else:
                    continue

            cycle_dict[e] = e_mapped

        return cycle_dict

    def __getitem__(self, x):
        return self._map.get(x, x)

    def __call__(self, x):
        return self._map.get(x, x)

    def __len__(self):
        '''Returns number of disjoint cycles of this permutation'''
        return len(self.decomp)

    def canonical_cycle_decomp(self, cycle_decomp):
        '''
        This is the same as finding every single connected component of the graph
        '''
        if not hasattr(self, '_map'):
            raise RuntimeError('Need to create the permutation map before'\
                               'computing the cycle decomposition!')

        visited = set()
        to_visit = set(range(1, self._max + 1))
        cycle = []
        while to_visit:
            # dfs on min element of elements
            curr = fst = min(to_visit)
            current_cycle = []

            while to_visit:
                current_cycle.append(curr)
                to_visit.remove(curr)
                #if curr in to_visit:
                #    to_visit.remove(curr)
                nxt = self.__getitem__(curr)
                if nxt == fst:
                    break
                curr = nxt

            cycle.append(current_cycle)

        return canonicalize(cycle)

    def inverse(self):
        # should this return another object?
        # the list of reversed perms?
        inv = map(lambda x: x[::-1], self.decomp)
        inv = canonicalize(inv)
        return Perm(inv)

    def __mul__(self, other):
        # rely on canonicalization to reduce the concatenation of the cycles
        prod = Perm(self.cycle_decomposition + other.cycle_decomposition)
        return prod

    def __repr__(self):
        if self._str is None:
            repr_str = ''
            for cyc in self.decomp:
                if len(cyc) <= 1:
                    continue
                if len(repr_str) == 0:
                    repr_str += (str(cyc))
                else:
                    repr_str += (', ' + str(cyc))

            self._str = repr_str

        # TODO: identity permutation gets rendered as (). Might not be ideal.
        if len(self._str) == 0:
            return '()'

        return self._str

    def order(self):
        # get lcm of cycle lengths
        cycle_lengths = [len(x) for x in self.cycle_decomposition]
        return lcm(*cycle_lengths)

    def matrix(self, n=None):
        if n is None:
            n = self._max
        p_matrix = np.zeros((n, n))

        # j->i   ===> p[i, j] = 1
        for j, i in self._map.items():
            # elements are 1-indexed
            p_matrix[i-1, j-1] = 1

        return p_matrix

def sn(n):
    if n in SN_CACHE:
        return SN_CACHE[n]
    perm_lsts = permutations(range(1, n+1))
    perms = [Perm.from_lst(lst) for lst in perm_lsts]
    SN_CACHE[n] = perms
    return perms

if __name__ == '__main__':
    group = sn(int(sys.argv[1]))
    print(len(group))
    for p in group:
        print(p)
