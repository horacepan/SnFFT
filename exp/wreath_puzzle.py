import pdb
import numpy as np
import random
from group_puzzle import GroupPuzzle

PYRAMINX_IDENT = ((0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6))
PYRAMINX_GENERATORS = [
    ((0, 0, 1, 1, 0, 0), (1, 3, 4, 2, 5, 6)),
    ((1, 1, 0, 0, 0, 0), (2, 6, 3, 4, 5, 1)),
    ((1, 0, 1, 0, 0, 0), (5, 2, 1, 4, 3, 6)),
    ((0, 0, 0, 1, 1, 0), (1, 2, 3, 5, 6, 4)),
    ((0, 1, 1, 0, 0, 0), (1, 4, 2, 3, 5, 6)),
    ((1, 0, 0, 0, 0, 1), (6, 1, 3, 4, 5, 2)),
    ((0, 0, 1, 0, 1, 0), (3, 2, 5, 4, 1, 6)),
    ((0, 0, 0, 1, 0, 1), (1, 2, 3, 6, 4, 5))
]

def px_perm_inv(p):
    return tuple(p.index(i) + 1 for i in range(1, len(p) + 1))

def px_perm_dot(perm, tup):
    pinv = px_perm_inv(perm)
    return (
        tup[pinv[0]-1],
        tup[pinv[1]-1],
        tup[pinv[2]-1],
        tup[pinv[3]-1],
        tup[pinv[4]-1],
        tup[pinv[5]-1]
    )

def px_cyc_add(c1, c2, n):
    return tuple(
        (c1[i] + c2[i]) % n for i in range(len(c1))
    )

def px_mul(p1, p2):
    return tuple(p1[p2[i] - 1] for i in range(len(p1)))

def px_wreath_mul(c1, p1, c2, p2, n):
    p1_dot_c2 = px_perm_dot(p1, c2)
    ori = px_cyc_add(c1, p1_dot_c2, n)
    perm = px_mul(p1, p2)
    return ori, perm

class WreathPuzzle(GroupPuzzle):
    def __init__(self):
        self.generators = None
        self._start_states = None
        self._ss_set = None
        self.cyc_size = None

    def num_nbrs(self):
        return len(self.generators)

    def group_mult(self, a, b):
        aor, aperm = a
        bor, bperm = b
        ori, perm = px_wreath_mul(aor, aperm, bor, bperm, 2)
        return (ori, perm)

    def moves(self, tup):
        return self.generators

    def nbrs(self, tup):
        output = []
        for g in self.moves(tup):
            output.append(self.group_mult(g, tup))
        return [self.group_mult(g, tup) for g in self.moves(tup)]

    def is_done(self, tup):
        return tup in self._ss_set

    def start_states(self):
        return self._start_states

    def random_walk(self, length):
        s = random.choice(self._start_states)
        states = [s]
        for _ in range(length - 1):
            s = self.random_step(s)
            states.append(s)

        return states

    def scramble(self, state, length):
        for _ in range(length):
            action = np.random.choice(len(self.generators))
            state = self.step(state, action)
        return state

    def random_state(self, length):
        state = random.choice(self._start_states)
        return self.scramble(state, length)

    def step(self, tup, action):
        o, p = tup
        ot, pt = self.generators[action]
        return px_wreath_mul(ot, pt, o, p, self.cyc_size)

    def random_step(self, tup):
        move = random.choice(self.moves(tup))
        return self.group_mult(move, tup)

    def random_move(self, state=None):
        return np.random.choice(len(self.generators))


class Pyraminx(WreathPuzzle):
    def __init__(self):
        self.generators = PYRAMINX_GENERATORS
        self._start_states = [PYRAMINX_IDENT]
        self._ss_set = set(self._start_states)
        self.cyc_size = 2

class Cube2(WreathPuzzle):
    def __init__(self):
        self.generators = CUBE2_GENERATORS
        self._start_states = [CUBE2_STARTS]
        self._ss_set = set(self._start_states)
        self.cyc_size = 3

    def render(self, state):
        pass
