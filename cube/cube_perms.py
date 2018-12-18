import time
import pdb
from str_cube import *

def rot_permutations(cube):
    '''
    cube: a length 24 cube string representation of the 2x2 cube
    Returns a list of all the cubes(given as strings) that result from the
    rigid isometries of the input cube
    '''
    perm_strs = [cube]

    # rotate a cube about its 3 axes
    ax_rots = [
        rot_x(cube),
        rot_x(cube, 2),
        rot_x(cube, 3),
        rot_y(cube),
        rot_y(cube, 2),
        rot_y(cube, 3),
        rot_z(cube),
        rot_z(cube, 2),
        rot_z(cube, 3),
    ]
  
    # rotate the cube about the 6 midpoint connecting two diagonally opposite faces 
    midpt_rots = [
        rot_x(rot_z(rot_x(cube), 3)),
        rot_x(rot_z(rot_x(cube))),
        rot_x(rot_z(cube, 2)),
        rot_y(rot_z(rot_y(cube))),
        rot_z(rot_x(cube, 2), 3),
        rot_y(rot_z(cube), 2)
    ]

    # rotate the cube about the 4 diagonals of the cube
    diag_rots = [
        # back left to front right diag
        rot_z(rot_x(cube, 3)),
        rot_z(rot_x(rot_z(rot_x(cube, 3)), 3)),

        # front top right to bot left top
        rot_x(rot_y(cube)),  #!
        rot_x(rot_y(rot_x(rot_y(cube)))), 

        # front top left to bot back right
        rot_y(rot_x(cube, 3)),
        rot_y(rot_x(rot_y(rot_x(cube, 3)), 3)), #!

        # front bot left to back top right
        rot_x(rot_y(cube), 3),
        rot_x(rot_y(rot_x(rot_y(cube), 3)), 3)
    ]

    perm_strs.extend(ax_rots)
    perm_strs.extend(midpt_rots)
    perm_strs.extend(diag_rots)
    return perm_strs

def str_to_wreath(cube_str):
    cubie_map = {
        'f': ['fru', 'frd', 'fld', 'flu'],
        'b': ['blu', 'bld', 'brd', 'bru'],
        'r': ['rbu', 'rbd', 'rfd', 'rfu'],
        'l': ['lfu', 'lfd', 'lbd', 'lbu'],
        'u': ['urb', 'urf', 'ulf', 'ulb'],
        'd': ['drf', 'drb', 'dlb', 'dlf'],
    }

    # need to map the cubies to orientations.
    # fru -> 0, fur -> 1, etc
    color_map: {

    }

def test_permutations():
    cube = init_2cube()
    perms = rot_permutations(cube)
    assert len(set(perms)) == 24

    for s in perms:
        for f in FACES:
            assert len(set(get_face(s, f))) == 1

def time_perms(n):
    c = init_2cube()
    cubes = [scramble(c, 100) for _ in range(n)]

    start = time.time()
    all_cubes = []
    for c in cubes:
        all_cubes.extend(rot_permutations(c))
    end = time.time()
    print('n = {:7s} | Isometry computation time {:.2f}'.format(str(n), end-start))

if __name__ == '__main__':
    for n in [10, 100, 1000, 10000, 100000]:
        time_perms(n)
    test_permutations()
