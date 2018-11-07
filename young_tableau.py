import os
import psutil
import resource
import pdb
import time
import itertools
import numpy as np
from functools import total_ordering
from utils import check_memory

def swap(x, i, j):
    if x == i:
        return j
    elif x == j:
        return i
    else:
        return x

class FerrersDiagram:
    def __init__(self, partition):
        '''
        partition: tuple of ints
        '''
        self.partition = partition
        self.size = sum(partition)

    def branch_up(self):
        pass

    def branch_down(self):
        pass

    def gen_tableaux(self, perms=None):
        '''
        Returns: list of the filled in
        '''
        tabs = []

        # 1 is always the top left corner so no need to do this
        #for p in itertools.permutations(range(2, self.size+1)):
        if perms is None:
            perms = ((1, ) + p for p in itertools.permutations(range(2, self.size+1)))

        for p in perms:
            if YoungTableau.valid_static(self.partition, p):
                yt = make_young_tableau(self.partition, p)
                tabs.append(yt)
        return tabs

def make_young_tableau(shape, vals):
    '''
    shape: tuple of row size of ferrers diagram. Sum of tuple elements is n
    vals: a permutation of {1, 2, ..., n}
    Return a filled in Young Tableau

    Ex: shape = (2, 1), vals = [1, 2, 3]
        Returns: [1][2]
                 [3]

        shape = (2, 2), vals = [1, 3, 2, 4]
        Returns: [1][3]
                 [2][4]
    '''
    contents = []
    i = 0
    for rowsize in shape:
        contents.append(vals[i:i+rowsize])
        i += rowsize
    return YoungTableau(contents, vals)

@total_ordering
class YoungTableau:
    CACHE = {}
    def __init__(self, contents, vals=None):
        '''
        contents: list of tuples of the filled in ferrers blocks
        vals: the full tuple for the permutation
        '''
        self.partition = tuple(map(lambda x: len(x), contents))
        self.contents = contents
        self.size = sum(self.partition)
        self.vals = vals

        if self.partition not in YoungTableau.CACHE:
            YoungTableau.CACHE[self.partition] = {}
        YoungTableau.CACHE[self.partition][vals] = self

    def __lt__(self, other):
        '''
        Use last letter ordering to determine which tableau is "larger"
        self is less than other if element n is on a higher row.
        if n is on the same row then check n-1, and so on.
        '''
        for i in range(self.size, 0, -1):
            # get row of i in self and in other
            if self.get_row(i) < other.get_row(i):
                return True
            elif self.get_row(i) > other.get_row(i):
                return False
            # otherwise they're in the same row and you proceed
        return False

    # this should be a static/class function
    @staticmethod
    def valid_static(shape, contents):
        i = 0
        rows = []
        curr_row_len = shape[0]
        prev_row_len = None
        curr_row_idx = 0
        row_idx = 0

        for i in range(len(contents)):
            if row_idx == curr_row_len:
                # start of a new row
                curr_row_idx += 1
                prev_row_len = curr_row_len
                curr_row_len = shape[curr_row_idx]
                row_idx = 0
            else:
                if row_idx < curr_row_len and i > 0:
                    if contents[i] < contents[i-1]:
                        return False
            if prev_row_len is not None:
                if contents[i] < contents[i - prev_row_len]:
                    return False
            row_idx += 1
        return True

    def valid(self):
        '''
        Checks if this young tableaux is valid.
        IE: each row must be increasing. each col must be increasing.
        '''
        for row in self.contents:
            # assert increasing
            for i in range(1, len(row)):
                if row[i-1] > row[i]:
                    return False

        max_cols = len(self.contents[0])
        for c in range(max_cols):
            for row_idx in range(1, len(self.contents)):
                row = self.contents[row_idx]
                if len(row) <= c:
                    break

                prev_row = self.contents[row_idx - 1]
                if prev_row[c] > row[c]:
                    return False
        return True

    '''
    def get_row_col(self, val):
        idx = self.contents.index(val)
        # see where this index lines up
        cnt = 0
        for row_idx, row_len in enumerate(self.partition):
            if idx < cnt + row_len
                row_idx = 
            cnt += row_len
    '''

    def get_row(self, val):
        '''
        Return the row number (1-indexed) of val
        If val is not in the tableaux (bigger than size or less than 1), return None
        '''
        for row_idx, row in enumerate(self.contents):
            if val in row:
                return row_idx + 1
        return None

    def get_col(self, val):
        '''
        Return the row number (1-indexed) of val
        If val is not in the tableaux (bigger than size or less than 1), return None
        '''
        for c in range(len(self.contents[0])):
            col = [row[c] for row in self.contents if len(row) > c]
            if val in col:
                return c + 1
        return None

    def __repr__(self):
        '''
        Pretty prints the Young Tableau
        '''
        rep_str = ''
        for idx, l in enumerate(self.contents):
            rep_str += ''.join(map(lambda x: '[{}]'.format(x), l))
            if idx != len(self.contents) - 1:
                rep_str += '\n'
        return rep_str


    def transpose(self, transposition):
        '''
        Returns the Young Tableau you'd get by applying the transposition to this tableau if
        the resulting tableaux is valid. Otherwise, return None
        '''
        i, j = transposition
        swapped = tuple(swap(k, i, j) for k in self.vals)
        return YoungTableau.CACHE[self.partition].get(swapped, None)

def benchmark():
    '''
    Benchmark time/memory usage for generating all YoungTableau for S_8
    '''
    shapes = [
        (8,),
        (7, 1),
        (6, 2),
        (6, 1, 1),
        (5, 3),
        (5, 2, 1),
        (5, 1, 1, 1),
        (4, 4),
        (4, 3, 1),
        (4, 2, 2),
        (4, 2, 1, 1),
        (4, 1, 1, 1, 1),
        (3, 3, 2),
        (3, 3, 1, 1),
        (3, 2, 2, 1),
        (3, 2, 1, 1, 1),
        (3, 1, 1, 1, 1, 1),
        (2, 2, 2, 2),
        (2, 2, 2, 1, 1),
        (2, 2, 1, 1, 1, 1),
        (2, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ]
    total_tabs = 0
    tstart = time.time()
    perms = list(((1, ) + p for p in itertools.permutations(range(2, 8+1))))
    for shape in shapes:
        start = time.time()
        f = FerrersDiagram(shape)
        tabs = f.gen_tableaux(perms)
        total_tabs += len(tabs)
        done = time.time() - start
        #print('Time to create {:5} tableaux for partition {:25} : {:.3f}'.format(len(tabs), str(shape), done))
        #print('-' * 80)
    print('Total tabs for partitions of 8: {}'.format(total_tabs))
    tend = time.time() - tstart
    print('Total time to find all tableaux: {:3f}'.format(tend))

    cache_size = 0
    mats = {}
    for k, v in YoungTableau.CACHE.items():
        cache_size += len(v)
        mats[k] = np.random.random((len(v), len(v)))
    print('Cache size: {}'.format(cache_size))
    #check_memory()

    v1 = (1, 2,3,4,5,6,7,8)
    v2 = (1, 2,3,4,5,6,8,7)
    shape = (7, 1)
    cache = YoungTableau.CACHE
    yt1 = cache[shape][v1]
    yt2 = cache[shape][v2]
    print(id(yt1.transpose((7,8))) == id(yt2))
    print(id(yt2.transpose((7,8))) == id(yt1))
    pdb.set_trace()
if __name__ == '__main__':
    benchmark()
