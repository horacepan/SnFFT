import sys
import resource
import pickle
import time
import pdb
from str_cube import *
from collections import deque
from itertools import permutations

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        PREFIX = '/{}/hopan/test/'.format(sys.argv[1])
    else:
        PREFIX = '/{}/hopan/'.format(sys.argv[1])
else:
    PREFIX = '/local/hopan/'

SAVE_PATH = PREFIX + 'cube_dist_full.txt'
print('Save path: {}'.format(SAVE_PATH))
PICKLE_PATH = PREFIX + 'cube_dist_full.pkl'

def bfs(root):
    root_perms = perms(root)
    dist_dict = {p: 0 for p in root_perms}
    to_visit = deque([(root, 0)])
    iters = 0
    while to_visit:
        curr, dist = to_visit.popleft()

        for nbr in neighbors(curr):
            if nbr not in dist_dict:
                to_visit.append((nbr, dist + 1))
                dist_dict[nbr] = dist + 1

        if 'test' in PREFIX and iters > 6:
           break
        iters += 1
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
    Given a permutation of S_6, return the corresponding permuted cube string.
    Note this is only valid for the initial 2-cube.
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

    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Consumed {:.2f}mb memory".format(res_/(1024**2)))

if __name__ == '__main__':
    test()
