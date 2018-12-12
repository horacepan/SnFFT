import random
from cube import Cube
import pdb
'''
The contains a barebones implementation of manipulating a 2x2 Rubiks Cube.
'''

#FACES = ['u', 'd',  'r', 'l', 'f', 'b']
#COLORS = ['G', 'B', 'M', 'R', 'W', 'Y']
FACES = ['u', 'd',  'l', 'r', 'f', 'b']
ALL_ROTS = [
    'u', 'd',  'l', 'r', 'f', 'b',
    'iu', 'id',  'il', 'ir', 'if', 'ib'
]
FACE_INDEX = {f: idx for idx, f in enumerate(FACES)}
COLORS = ['G', 'B', 'R', 'M', 'W', 'Y']
def init_2cube():
    cube = ''
    for c, f in zip(COLORS, FACES):
        cube = cube + (4 * c)
    return cube

def get_face(cube_str, f):
    assert f in FACES
    f_idx = FACE_INDEX[f]
    return cube_str[f_idx * 4: f_idx * 4 + 4]

def rotate(str_cube, f):
    rot_func = eval('rot_{}'.format(f))
    rot_func(str_cube)
    return rot_func(str_cube)

def swap_faces(cube_str, face_cycles):
    '''
    cube_str: string representing a cube
    face_cycles: list of tuples of faces (letters in FACES)
    Ex:
        swap_faces(init_2cube(), [('u', 'd'), ('l', 'r')])
    '''
    new_str = ''
    new_cube = {}
    for faces in face_cycles:
        for i in range(len(faces)):
            new_cube[faces[i]] = get_face(cube_str, faces[(i-1) % len(faces)])

    for f in FACES:
        if f not in new_cube:
            new_cube[f] = get_face(cube_str, f)

    return make_cube_str_dict(new_cube)

def cycle(s):
    '''
    s: length 4 string representing the face of a 2x2 cube
        s[0]s[1]
        s[2]s[3]
    Returns: A string representing a 90 degree clockwise rotation of the input face
    '''
    return '{}{}{}{}'.format(
        s[2], s[0], s[3], s[1]
    )

def cycle_cc(s):
    '''
    s: length 4 string representing the face of a 2x2 cube
        s[0]s[1]
        s[2]s[3]
    Returns: A string representing a 90 degree clockwise rotation of the input face
    '''

    return '{}{}{}{}'.format(
        s[1], s[3], s[0], s[2]
    )

def cycle_180(s):
    '''
    s: length 4 string representing the face of a 2x2 cube
    Given a face:    s[0]s[1]
                     s[2]s[3]
    Return the face: s[3][s2] which is represented as a 1-d string: s[3]s[2]s[1]s[0]
                     s[1]s[0]
    '''
    return '{}{}{}{}'.format(
        s[3], s[2], s[1], s[0]
    )

def make_cube_str(u, d, l, r, f, b):
    return '{}{}{}{}{}{}'.format(u,d,l,r,f,b)

def make_cube_str_dict(face_dict):
    return '{}{}{}{}{}{}'.format(
        face_dict['u'],
        face_dict['d'],
        face_dict['l'],
        face_dict['r'],
        face_dict['f'],
        face_dict['b']
    )

def neighbors(cube_str):
    nbrs = [
        rot_u(cube_str),
        rot_d(cube_str),
        rot_l(cube_str),
        rot_r(cube_str),
        rot_f(cube_str),
        rot_b(cube_str),
        rot_iu(cube_str),
        rot_id(cube_str),
        rot_il(cube_str),
        rot_ir(cube_str),
        rot_if(cube_str),
        rot_ib(cube_str)
    ]
    return nbrs

def neighbors_fixed_core(cube_str):
    '''
    This should give you 24x fewer states due to the isometries of the cube.
    '''
    nbrs = [
        rot_u(cube_str),
        rot_d2(cube_str),
        rot_l2(cube_str),
        rot_r(cube_str),
        rot_f(cube_str),
        rot_b2(cube_str),
        rot_iu(cube_str),
        rot_id2(cube_str),
        rot_il2(cube_str),
        rot_ir(cube_str),
        rot_if(cube_str),
        rot_ib2(cube_str)
    ]
    return nbrs

def render(cube_str):
    # TODO: refactor the cube render function
    cube = Cube.from_str(cube_str)
    cube.render()

# TODO: this is hardcoded for 2x2. Should make it generic for nxn
# Assumption: rotating u will keep the "cores" of l/b/r/f the same
def rot_u(cube_str):
    _f = get_face(cube_str, 'f')
    _l = get_face(cube_str, 'l')
    _b = get_face(cube_str, 'b')
    _r = get_face(cube_str, 'r')

    u_face = cycle(get_face(cube_str, 'u'))
    d_face = get_face(cube_str, 'd')
    l_face =  _f[:2] + _l[2:]
    b_face =  _l[:2] + _b[2:]
    r_face =  _b[:2] + _r[2:]
    f_face =  _r[:2] + _f[2:]
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )


def rot_iu(cube_str):
    _f = get_face(cube_str, 'f')
    _l = get_face(cube_str, 'l')
    _b = get_face(cube_str, 'b')
    _r = get_face(cube_str, 'r')

    u_face = cycle_cc(get_face(cube_str, 'u'))
    d_face = get_face(cube_str, 'd')
    l_face =  _b[:2] + _l[2:] # b's top becomes l's top
    b_face =  _r[:2] + _b[2:] # r's top becomes b's top
    r_face =  _f[:2] + _r[2:] # f's top becomes r's top
    f_face =  _l[:2] + _f[2:] # l's top becomes f's top

    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_d(cube_str):
    _f = get_face(cube_str, 'f')
    _l = get_face(cube_str, 'l')
    _b = get_face(cube_str, 'b')
    _r = get_face(cube_str, 'r')

    u_face = get_face(cube_str, 'u')
    d_face = cycle(get_face(cube_str, 'd'))
    l_face =  _l[:2] + _f[2:]
    b_face =  _b[:2] + _l[2:]
    r_face =  _r[:2] + _b[2:]
    f_face =  _f[:2] + _r[2:]
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_id(cube_str):
    _f = get_face(cube_str, 'f')
    _l = get_face(cube_str, 'l')
    _b = get_face(cube_str, 'b')
    _r = get_face(cube_str, 'r')

    u_face = get_face(cube_str, 'u')
    d_face = cycle_cc(get_face(cube_str, 'd'))
    l_face =  _l[:2] + _b[2:] # b's bot becomes l's bot
    b_face =  _b[:2] + _r[2:] # r's bot becomes b's bot
    r_face =  _r[:2] + _f[2:] # f's bot becomes r's bot
    f_face =  _f[:2] + _l[2:] # l's bot becomes f's bot
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_d2(cube_str):
    return rot_iu(cube_str)

def rot_id2(cube_str):
    return rot_u(cube_str)

# Assumption: the f/u/b/d "cores" stay in place. Every one keeps their left except b, which
# keeps it's right(due to reversal)
def rot_r(cube_str):
    _f = get_face(cube_str, 'f')
    _u = get_face(cube_str, 'u')
    _b = get_face(cube_str, 'b')
    _d = get_face(cube_str, 'd')

    f_face = _f[0] + _d[1] + _f[2] + _d[3] # d's right becomes f's right
    u_face = _u[0] + _f[1] + _u[2] + _f[3] # f's right becomes u's right
    b_face = _u[3] + _b[1] + _u[1] + _b[3] # u's right(rev) becomes b's left
    d_face = _d[0] + _b[2] + _d[2] + _b[0] # b's left(rev) becomes d's right

    r_face = cycle(get_face(cube_str, 'r'))
    l_face = get_face(cube_str, 'l')
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_ir(cube_str):
    _f = get_face(cube_str, 'f')
    _u = get_face(cube_str, 'u')
    _b = get_face(cube_str, 'b')
    _d = get_face(cube_str, 'd')

    f_face = _f[0] + _u[1] + _f[2] + _u[3] # u's right becomes f's right
    u_face = _u[0] + _b[2] + _u[2] + _b[0] # b's left(rev) becomes u's right
    b_face = _d[3] + _b[1] + _d[1] + _b[3] # d's right(rev) becomes b's left
    d_face = _d[0] + _f[1] + _d[2] + _f[3] # f's right(rev) becomes d's right

    r_face = cycle_cc(get_face(cube_str, 'r'))
    l_face = get_face(cube_str, 'l')
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_l(cube_str):
    _f = get_face(cube_str, 'f')
    _u = get_face(cube_str, 'u')
    _b = get_face(cube_str, 'b')
    _d = get_face(cube_str, 'd')

    u_face = _f[0] + _u[1] + _f[2] + _u[3] # f's left becomes u's left
    f_face = _d[0] + _f[1] + _d[2] + _f[3] # d's left becomes f's left
    d_face = _b[3] + _d[1] + _b[1] + _d[3] # b's right(rev) becomes d's left
    b_face = _b[0] + _u[2] + _b[2] + _u[0] # u's left(rev) becomes b's right

    r_face = get_face(cube_str, 'r')
    l_face = cycle_cc(get_face(cube_str, 'l'))
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_il(cube_str):
    _f = get_face(cube_str, 'f')
    _u = get_face(cube_str, 'u')
    _b = get_face(cube_str, 'b')
    _d = get_face(cube_str, 'd')

    u_face = _b[3] + _u[1] + _b[1] + _u[3] # b's right(rev) becomes u's left
    f_face = _u[0] + _f[1] + _u[2] + _f[3] # u's left becomes f's left
    d_face = _f[0] + _d[1] + _f[2] + _d[3] # f's left becomes d's left
    b_face = _b[0] + _d[2] + _b[2] + _d[0] # d's left(rev) becomes b's right

    r_face = get_face(cube_str, 'r')
    l_face = cycle(get_face(cube_str, 'l'))
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_l2(cube_str):
    return rot_ir(cube_str)

def rot_il2(cube_str):
    return rot_r(cube_str)


# Assumption: the u/r/d/l "cores" stay in place
def rot_f(cube_str):
    _u = get_face(cube_str, 'u')
    _r = get_face(cube_str, 'r')
    _d = get_face(cube_str, 'd')
    _l = get_face(cube_str, 'l')

    u_face = _u[0] + _u[1] + _l[3] + _l[1] # l's right(rev) becomes u's bot
    r_face = _u[2] + _r[1] + _u[3] + _r[3] # u's bot becomes r's left
    d_face = _r[2] + _r[0] + _d[2] + _d[3] # r's left (rev) becomes d's top
    l_face = _l[0] + _d[0] + _l[2] + _d[1] # d's top becomes l's right

    f_face = cycle(get_face(cube_str, 'f'))
    b_face = get_face(cube_str, 'b')
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_if(cube_str):
    _u = get_face(cube_str, 'u')
    _r = get_face(cube_str, 'r')
    _d = get_face(cube_str, 'd')
    _l = get_face(cube_str, 'l')

    u_face = _u[0] + _u[1] + _r[0] + _r[2] # r's left becomes u's bot
    r_face = _d[1] + _r[1] + _d[0] + _r[3] # d's top(rev) becomes r's left
    d_face = _l[1] + _l[3] + _d[2] + _d[3] # l's right becomes d's top
    l_face = _l[0] + _u[3] + _l[2] + _u[2] # u's bot(rev) becomes l's right

    f_face = cycle_cc(get_face(cube_str, 'f'))
    b_face = get_face(cube_str, 'b')
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_b(cube_str):
    _u = get_face(cube_str, 'u')
    _r = get_face(cube_str, 'r')
    _d = get_face(cube_str, 'd')
    _l = get_face(cube_str, 'l')

    u_face = _r[1] + _r[3] + _u[2] + _u[3] # r's right becomes u's top
    l_face = _u[1] + _l[1] + _u[0] + _l[3] # u's top(rev) becomes l's left
    d_face = _d[0] + _d[1] + _l[0] + _l[2] # l's left becomes d's bot
    r_face = _r[0] + _d[3] + _r[2] + _d[2] # d's bot(rev) becomes r's right

    f_face = get_face(cube_str, 'f')
    b_face = cycle(get_face(cube_str, 'b'))
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_ib(cube_str):
    _u = get_face(cube_str, 'u')
    _r = get_face(cube_str, 'r')
    _d = get_face(cube_str, 'd')
    _l = get_face(cube_str, 'l')

    u_face = _l[2] + _l[0] + _u[2] + _u[3] # l's left(rev) becomes u's top
    l_face = _d[2] + _l[1] + _d[3] + _l[3] # d's bot becomes l's left
    d_face = _d[0] + _d[1] + _r[3] + _r[1] # r's right becomes d's bot
    r_face = _r[0] + _u[0] + _r[2] + _u[1] # u's top becomes r's right

    f_face = get_face(cube_str, 'f')
    b_face = cycle_cc(get_face(cube_str, 'b'))
    return make_cube_str(
        u_face,
        d_face,
        l_face,
        r_face,
        f_face,
        b_face
    )

def rot_b2(cube_str):
    return rot_f(cube_str)

def rot_ib2(cube_str):
    return rot_if(cube_str)

def rot_z(cube_str, times=1):
    '''Rotate entire cube clockwise about the z axis'''
    for _ in range(times):
        cube_str = rot_u(rot_d(cube_str))
    return cube_str

def rot_x(cube_str, times=1):
    '''Rotate entire cube clockwise about the x axis(through the left/right faces'''
    for _ in range(times):
        cube_str = rot_l(rot_r(cube_str))
    return cube_str

def rot_y(cube_str, times=1):
    '''Rotate entire cube clockwise about the y axis(through the f/b faces)
    (front face on top)
    '''
    for _ in range(times):
        cube_str = rot_f(rot_ib(cube_str))
    return cube_str

def check_same():
    # Check that rotating the cube doesnt do anything
    cube = init_2cube()
    assert cube == rot_u(rot_d2(cube))
    assert cube == rot_d2(rot_u(cube))
    assert cube == rot_iu(rot_id2(cube))
    assert cube == rot_id2(rot_iu(cube))
    print('u/d okay')

    assert cube == rot_r(rot_l2(cube))
    assert cube == rot_l2(rot_r(cube))
    assert cube == rot_ir(rot_il2(cube))
    assert cube == rot_il2(rot_ir(cube))
    print('l/r okay')

    assert cube == rot_f(rot_ib2(cube))
    assert cube == rot_ib2(rot_f(cube))
    assert cube == rot_if(rot_b2(cube))
    assert cube == rot_b2(rot_if(cube))
    print('f/b okay')

def scramble(cube, n=1):
    '''
    Performs n random moves to the input cube
    cube: a string representing the cube state
    n: int, number of random moves to make

    Returns a cube string.
    '''
    for _ in range(n):
        f = random.choice(ALL_ROTS)
        cube = rotate(cube, f)
    return cube

if __name__ == '__main__':
    pass
