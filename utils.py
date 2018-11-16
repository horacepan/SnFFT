import os
import psutil
import resource

def s1():
    permutations = [ [(1,)] ]
    return permutations

def s2():
    permutations = [
        [(1,)],
        [(1,2)],
    ]
    return permutations

def s3():
    permutations = [
        [(1,)],
        [(1, 2)],
        [(1, 3)],
        [(2, 3)],
        [(1, 2, 3)],
        [(1, 3, 2)],
    ]
    return permutations
def s4():
    permutations = [
        [(1,)],
        [(1, 2)],
        [(1, 3)],
        [(1, 4)],
        [(2, 3)],
        [(2, 4)],
        [(3, 4)],
        [(1, 2), (3, 4)],
        [(1, 3), (2, 4)],
        [(1, 4), (2, 3)],
        [(1, 2, 3)],
        [(1, 2, 4)],
        [(1, 3, 2)],
        [(1, 3, 4)],
        [(1, 4, 2)],
        [(1, 4, 3)],
        [(2, 3, 4)],
        [(2, 4, 3)],
        [(1, 2, 3, 4)],
        [(1, 2, 4, 3)],
        [(1, 3, 2, 4)],
        [(1, 3, 4, 2)],
        [(1, 4, 2, 3)],
        [(1, 4, 3, 2)],
    ]
    return permutations

def sn(n):
    '''
    Return the permutations (in cycle notation) for n = 1, 2, 3, 4
    Returns a list of list of tuples
    '''
    fname = 's{}'.format(n)
    func = eval(fname)
    return func()


def check_memory():
    res_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    process = psutil.Process(os.getpid())
    resp = process.memory_info().rss
    print("Consumed {:.2f}mb memory".format(res_/(1024**2)))
