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

    @staticmethod
    def nbrs(tup):
        return [px_mult(g, tup) for g in S8Puzzle.generators]

    @staticmethod
    def is_done(tup):
        return tup in S8Puzzle._ss_set

    @staticmethod
    def start_states():
        return S8Puzzle._start_states

    @staticmethod
    def random_walk(length):
        states = [random.choice(S8Puzzle._start_states)]
        s = states[0]
        for _ in range(length - 1):
            s = S8Puzzle.random_step(s)
            states.append(s)

        return states

    @staticmethod
    def scramble(state, length):
        for _ in range(length):
            action = np.random.choice(len(S8Puzzle.generators))
            state = S8Puzzle.step(state, action)
        return state

    @staticmethod
    def random_state(length=100):
        state = random.choice(S8Puzzle._start_states)
        return S8Puzzle.scramble(state, length)

    @staticmethod
    def random_step(tup):
        action = random.choice(S8Puzzle.generators)
        return px_mult(action, tup)

    @staticmethod
    def random_move():
        return np.random.choice(6)

    @staticmethod
    def step(tup, action):
        g = S8Puzzle.generators[action]
        return px_mult(g, tup)

if __name__ == '__main__':
    x = S8Puzzle.random_state()
    print(x)
