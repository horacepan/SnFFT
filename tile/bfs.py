import os
import sys
sys.path.append('../')
from utils import check_memory
import pdb
from collections import deque
from tile_env import TileEnv, neighbors

def np_to_tup(grid):
    return tuple(i for row in grid for i in row)

def tup_to_str(tup):
    return ''.join(str(i) for i in tup)

def np_to_str(grid):
    return tup_to_str(np_to_tup(grid))

def bfs(root, fname):
    print('Writing to: {}'.format(fname))
    with open(fname, 'w') as f:
        to_visit = deque([(root, 0)])
        dist_dict = {np_to_tup(root): 0} #{np_to_tup(root): 0}
        f.write('{},0\n'.format(np_to_str(root)))

        while to_visit:
            curr, dist = to_visit.popleft()
            ctup = np_to_tup(curr)

            for nbr in neighbors(curr).keys():
                ntup = np_to_tup(nbr)
                if ntup not in dist_dict:
                    dist_dict[ntup] = dist + 1
                    f.write('{},{}\n'.format(tup_to_str(ntup), dist + 1))
                    # append the grid not the nbr
                    to_visit.append((nbr, dist + 1))
    check_memory()
    return dist_dict

def main():
    n = int(sys.argv[1])
    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    else:
        prefix = '/local/hopan/tile/'
    fname = os.path.join(prefix, 'tile{}.txt'.format(n))
    print('Saving in: {}'.format(fname))
    start_state = TileEnv.solved_grid(n)
    res = bfs(start_state, fname)
    print('Num states: {}'.format(len(res)))

if __name__ == '__main__':
    main()
