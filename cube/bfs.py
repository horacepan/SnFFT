import pickle
import time
import pdb
from str_cube import *
from collections import deque

PREFIX = '/local/hopan/'
SAVE_PATH = PREFIX + 'cube_dist.txt'
PICKLE_PATH = PREFIX + 'cube_dist.pkl'

def bfs(root):
    dist_dict = {}
    to_visit = deque([(root, 0)])

    while to_visit:
        curr, dist = to_visit.popleft()

        if (curr not in dist_dict) or (curr in dist_dict and dist < dist_dict[curr]):
            dist_dict[curr] = dist

        for nbr in neighbors(curr):
            if nbr not in dist_dict:
                to_visit.append((nbr, dist + 1))

    return dist_dict

def write_dist(dist_dict, save_path):
    with open(save_path, 'w') as f:
        for k, v in dist_dict.items():
            f.write('{},{}\n'.format(k,v))

def pickle_dist(dist_dict, pickle_path):
    with open(pickle_path, 'wb') as fhandle:
        pickle.dump(dist_dict, fhandle)

def test():
    start = time.time()
    cube = init_2cube()
    dist_dict = bfs(cube)
    elapsed = time.time() - start
    print('BFS time: {:.2f}'.format(elapsed))

    w_start = time.time()
    write_dist(dist_dict, SAVE_PATH)
    pickle_dist(dist_dict, PICKLE_PATH)
    w_elapsed = time.time() - w_start
    print('Write time: {:.2f}'.format(w_elapsed))

if __name__ == '__main__':
    test()
