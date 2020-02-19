'''
Wrapper class for S8 puzzle utilities including:
- generating random states
- checking if a state is solved

nbrs
is_done
start_states
random_walk
scramble: moves a given state by the given length
random_state: starts from a random state and scrambles

'''
import random
import numpy as np
from utility import px_mult

class S8Puzzle:
    generators = [
        (8, 1, 3, 4, 5, 6, 2, 7),
        (2, 7, 3, 4, 5, 6, 8, 1),
        (2, 3, 4, 1, 5, 6, 7, 8),
        (4, 1, 2, 3, 5, 6, 7, 8),
        (1, 2, 3, 4, 7, 5, 8, 6),
        (1, 2, 3, 4, 6, 8, 5, 7)
    ]

    _start_states = [
        (1, 2, 3, 4, 5, 6, 7, 8),
        (8, 1, 4, 5, 6, 3, 2, 7),
        (7, 8, 5, 6, 3, 4, 1, 2),
        (2, 7, 6, 3, 4, 5, 8, 1),
        (2, 3, 4, 1, 8, 5, 6, 7),
        (1, 4, 5, 8, 7, 6, 3, 2),
        (8, 5, 6, 7, 2, 3, 4, 1),
        (7, 6, 3, 2, 1, 4, 5, 8),
        (3, 4, 1, 2, 7, 8, 5, 6),
        (4, 5, 8, 1, 2, 7, 6, 3),
        (5, 6, 7, 8, 1, 2, 3, 4),
        (6, 3, 2, 7, 8, 1, 4, 5),
        (4, 1, 2, 3, 6, 7, 8, 5),
        (5, 8, 1, 4, 3, 2, 7, 6),
        (6, 7, 8, 5, 4, 1, 2, 3),
        (3, 2, 7, 6, 5, 8, 1, 4),
        (4, 3, 6, 5, 8, 7, 2, 1),
        (5, 4, 3, 6, 7, 2, 1, 8),
        (6, 5, 4, 3, 2, 1, 8, 7),
        (3, 6, 5, 4, 1, 8, 7, 2),
        (8, 7, 2, 1, 4, 3, 6, 5),
        (7, 2, 1, 8, 5, 4, 3, 6),
        (2, 1, 8, 7, 6, 5, 4, 3),
        (1, 8, 7, 2, 3, 6, 5, 4)
    ]
    _ss_set = set(_start_states)

    def __init__(self):
        pass

    def num_nbrs(self):
        return len(self.generators)

    def nbrs(self, tup):
        return [px_mult(g, tup) for g in self.generators]

    def is_done(self, tup):
        return tup in self._ss_set

    def start_states(self):
        return self._start_states

    def random_walk(self, length):
        states = [random.choice(self._start_states)]
        s = states[0]
        for _ in range(length - 1):
            s = self.random_step(s)
            states.append(s)

        return states

    def scramble(self, state, length):
        for _ in range(length):
            action = np.random.choice(len(self.generators))
            state = self.step(state, action)
        return state

    def random_state(self, length=100):
        state = random.choice(self._start_states)
        return self.scramble(state, length)

    def random_step(self, tup):
        action = random.choice(self.generators)
        return px_mult(action, tup)

    def random_move(self):
        return np.random.choice(6)

    def step(self, tup, action):
        g = self.generators[action]
        return px_mult(g, tup)

    def all_states(self):
        return self.dist_dict.keys()

if __name__ == '__main__':
    x = S8Puzzle.random_state()
    print(x)
