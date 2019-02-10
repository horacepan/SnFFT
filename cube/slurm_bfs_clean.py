import sys
import resource
import pickle
import time
from datetime import datetime
import pdb
from str_cube import *
from collections import deque
from itertools import permutations
from utils import check_memory
import argparse

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        PREFIX = '/{}/hopan/test/'.format(sys.argv[1])
    else:
        PREFIX = '/{}/hopan/'.format(sys.argv[1])
else:
    PREFIX = '/scratch/hopan/'

#SAVE_PATH = PREFIX + 'cube_full_{}.txt'.format(str(datetime.now()).replace(' ', '_'))
#SAVE_PATH = sys.argv[1]
#print('Save path: {}'.format(SAVE_PATH))
#PICKLE_PATH = PREFIX + 'cube_sym_mod.txt'

def bfs(root):
    SAVE_PATH = sys.argv[1]
    #print('Save path: {}'.format(SAVE_PATH))
    with open(SAVE_PATH, 'w') as f:
        #root_perms = perms(root)
        root_perms = [root]
        #dist_dict = {p: 0 for p in root_perms}
        for p in root_perms:
            f.write('{},0\n'.format(p))

        cube_set = set(root_perms)
        to_visit = deque([(c, 0) for c in root_perms]) # basically a distributed bfs from each "solved" state
        iters = 0
        while to_visit:
            curr, dist = to_visit.popleft()

            for nbr in neighbors_fixed_core(curr):
                #if nbr not in dist_dict:
                if nbr not in cube_set:
                    to_visit.append((nbr, dist + 1))
                    f.write('{},{}\n'.format(nbr, dist + 1))

                    #dist_dict[nbr] = dist + 1
                    cube_set.add(nbr)

            if 'test' in PREFIX and iters > 6:
               break
            iters += 1
    #return dist_dict

def dist_bfs(test=False):
    c = init_2cube()
    dist_dict = {}

    # this should actually start with all possible
    to_visit = deque([])
    for cs in all_init_states():
        to_visit.append((cs, 0))
        dist_dict[cs] = 0

    while to_visit:
        curr, dist = to_visit.popleft()
        if curr in dist_dict:
            dist_dict[curr] = min(dist, dist_dict[curr])
        else:
            dist_dict[curr] = dist

        for nbr in neighbors(curr):
            if nbr not in dist_dict:
                to_visit.append((nbr, dist + 1))
        if test:
            if len(dist_dict) > 30:
                break

    return dist_dict

def valid_cube_perm(perm):
    '''
    Check if the given permutation (list of ints) is a valid cube permutation
    '''
    U = FACES.index('u')
    R = FACES.index('r')
    F = FACES.index('f')
    valid_urf = set(
        ['urf', 'ufl', 'ulb', 'ubr', 'dfr', 'dlf', 'dbl', 'drb']
    )
    t0 = perm[U] + perm[R] + perm[F]
    t1 = perm[F] + perm[U] + perm[R]
    t2 = perm[R] + perm[F] + perm[U]

    if  (t0 not in valid_urf) and (t1 not in valid_urf) and (t2 not in valid_urf):
        return False

    valid_order = {'u':'d', 'd':'u', 'l':'r', 'r':'l', 'f':'b', 'b':'f'}
    for i in [0, 2, 4]:
        # is i getting mapped properly
        if valid_order[perm[i]] != perm[i+1]:
            return False
        # is i+1 getting mapped properly
        if valid_order[perm[i+1]] != perm[i]:
            return False
        # now check the u/l/f

    return True

def perm_to_cube(p, cube_str=None):
    '''
    Given a permutation of S_6, return the corresponding
    '''
    if cube_str is None:
        cube_str = init_2cube()

    cube_dict = {}
    for idx, f in enumerate(FACES):
        cube_dict[f] = get_face(cube_str, p[idx])

    return make_cube_str_dict(cube_dict)

def perms(cube_str):
    '''
    This is only valid if you start at the 2 cube's initial state. For other cube states
    the cube faces will rotate which makes things much harder to compute.
    '''
    cube = init_2cube()
    s6 = permutations(FACES)
    valid_p = [p for p in s6 if valid_cube_perm(p)]
    perm_cube_strs = [perm_to_cube(p) for p in valid_p]

    return perm_cube_strs

def write_dist(dist_dict, save_path):
    with open(save_path, 'w') as f:
        for k, v in dist_dict.items():
            f.write('{},{}\n'.format(k,v))

def write_wreath(dist_dict, wreath_path):
    with open(wreath_path, 'w') as f:
        for k, v in dist_dict.items():
            otup, ptup = get_wreath(k)
            str_otup = str(otup)[1:-1].replace(', ', '')
            str_ptup = str(otup)[1:-1].replace(', ', '')
            f.write('{},{},{}\n'.format(str_otup, str_ptup,v))

def pickle_dist(dist_dict, pickle_path):
    with open(pickle_path, 'wb') as fhandle:
        pickle.dump(dist_dict, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    start = time.time()
    cube = init_2cube()
    dist_dict = bfs(cube)
    elapsed = time.time() - start
    print('BFS time: {:.2f}'.format(elapsed))

    #w_start = time.time()
    #write_dist(dist_dict, SAVE_PATH)
    #pickle_dist(dist_dict, PICKLE_PATH)
    #w_elapsed = time.time() - w_start
    #print('Write time: {:.2f}'.format(w_elapsed))

    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Consumed {:.2f}mb memory".format(res_/(1024**2)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do bfs.')
    parser.add_argument('--savepath', type=str, default='')
    parser.add_argument('--pklpath', type=str, default='')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wreathpath', type=str, default='')
    args = parser.parse_args()

    print('Starting bfs')
    dist_dict = dist_bfs(args.test)
    print('Done with bfs')
    print('Size dict: {}'.format(len(dist_dict)))
    check_memory()

    if args.savepath:
        write_dist(dist_dict, args.savepath)
    if args.pklpath:
        pickle_dist(dist_dict, args.pklpath)
    if args.wreathpath:
        write_wreath(dist_dict, args.wreathpath)
    print('Done writing results')
    check_memory()
