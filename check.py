import os
import pdb
from utils import load_sparse_pkl
from utils import partition_parts, check_memory
import time

alphas = [
        (2, 3, 3),
        (4, 2, 2),
        (3, 1, 4),
        (1, 2, 5),
        (0, 4, 4),
        (5, 0, 3),
        (6, 1, 1),
        (2, 0, 6),
        (0, 1, 7),
        (8, 0, 0),
    ]

for a in alphas:
    print('alpha: {}'.format(a))
    start = time.time()
    for p in partition_parts(a):
        if os.path.exists('/local/hopan'):
            fname = '/local/hopan/cube/pickles_sparse/{}/{}.pkl'.format(a, p)
        elif os.path.exists('/scratch/hopan'):
            fname = '/scratch/hopan/cube/pickles_sparse/{}/{}.pkl'.format(a, p)
        elif os.path.exists('/project2/risi'):
            fname = '/project2/risi/cube/pickles_sparse/{}/{}.pkl'.format(a, p)

        try:
            load_sparse_pkl(fname)
            check_memory()
        except:
            pdb.set_trace()
            print('Unable to load {} | {}'.format(a, p))

    print('Done with {} | {:.2f}s'.format(a, time.time() - start))
    print('=' * 80)
