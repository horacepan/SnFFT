from utils import canonicalize, lcm
import numpy as np
import pdb

class Perm:
    '''
    Assume the cycle decomposition given is in canonical form(no repeated things).
    Not necessarily in simplest form.
    '''
    def __init__(self, cyc_decomp):
        self._str = None
        self._max = max(map(lambda z: max(z), cyc_decomp)) # might not be accurate
        self._map = self._init_map(cyc_decomp)
        self.decomp = self.canonical_cycle_decomp(cyc_decomp)

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

if __name__ == '__main__':
    p = Perm([(1,2,3), (5, 6)])
    print(p._map)
    print(p.matrix())
