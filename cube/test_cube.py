import time
import pdb
from str_cube import *
from cube import *

def test():
    cube = Cube(2)

    cube.random_step(20)
    faces = ['u', 'd', 'l', 'r', 'f', 'b']
    faces = ['iu', 'id', 'ir', 'il', 'if', 'ib']
    # faces: u, d, l, r, f, b
    for f in faces:
        for _ in range(10):
            str_cube = cube.str_state()
            if 'i' in f:
                cube.inv_rotate(f[1])
            rotated = rotate(str_cube, f)
            if (rotated != cube.str_state()):
                print('Incorrect for face: {}'.format(f))
                cube.render()
                print('====')
                rotated_cube = Cube.from_str(rotated)
                rotated_cube.render()
                exit()

            cube.random_step(1)

    print('All good!')

def benchmark(k=1000):
    '''
    Compare cube manipulation using Cube object versus pure string cube implementation
    '''
    cube = Cube(2)
    cube.random_step(20)
    str_cube = cube.str_state()

    c_s = time.time()
    for _ in range(k):
        cube.inv_rotate('u')
        cube.inv_rotate('d')
        cube.inv_rotate('l')
        cube.inv_rotate('r')
        cube.inv_rotate('f')
        cube.inv_rotate('b')

    t_cube = time.time() - c_s

    s_s = time.time()
    for _ in range(k):
        str_cube = rot_iu(str_cube)
        str_cube = rot_id(str_cube)
        str_cube = rot_il(str_cube)
        str_cube = rot_ir(str_cube)
        str_cube = rot_if(str_cube)
        str_cube = rot_ib(str_cube)


    t_s = time.time() - s_s
    print('Computing {} face moves'.format(k))
    print('Cube time: {:.2f}'.format(t_cube))
    print(' Str time: {:.2f}'.format(t_s))

if __name__ == '__main__':
    test()
    benchmark(1000)
