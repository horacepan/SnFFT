from collections import Counter
from itertools import product

def cube2_orientations():
    for tup in product(*[(0, 1, 2) for _ in range(8)]):
        if (sum(tup) % 3 == 0):
            yield tup
