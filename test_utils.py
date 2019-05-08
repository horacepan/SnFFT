import unittest
from utils import *
from complex_utils import *
from gen_sparse import convert_idx, block_indices
import numpy as np
import torch

def make_complex(A, B):
    Z = np.zeros(A.shape, dtype=np.complex128)
    Z.real = A
    Z.imag = B
    return Z

class TestUtils(unittest.TestCase):
    def test_partitions(self):
        n = 5
        parts = set(partitions(n))
        expected = [(1,1,1,1,1), (2,1,1,1), (2,2,1), (3,1,1), (3,2), (4,1), (5,)]
        self.assertEqual(len(parts), len(expected))

        for p in expected:
            self.assertTrue(p in parts)

    def test_weak_partitions(self):
        n = 6
        k = 2
        parts = set([tuple(x) for x in weak_partitions(n, k)])
        expected_parts = [
            (6,0), (5,1), (2, 4), (3,3),
            (0,6), (1,5), (4, 2)
        ]

        self.assertEqual(len(parts), len(expected_parts))
        for p in expected_parts:
            self.assertTrue(p in parts)

    def test_idx_convert(self):
        idx = (6, 1)
        shape = 3
        cols1 = 21
        cols2 = 1
        exp1 = (0, 19)
        exp2 = (19, 0)
        self.assertEqual(convert_idx(idx, shape, cols1), exp1)
        self.assertEqual(convert_idx(idx, shape, cols2), exp2)

        idx = (3, 4)
        shape = 13
        cols1 = 5
        cols2 = 6
        exp1 = (8, 3)
        exp2 = (7, 1)
        self.assertEqual(convert_idx(idx, shape, cols1), exp1)
        self.assertEqual(convert_idx(idx, shape, cols2), exp2)

    def test_block_indices(self):
        idx = (1, 4)
        block_size = 3
        output = block_indices(idx, block_size)
        expected_indices = [
            (3, 12), (3, 13), (3, 14),
            (4, 12), (4, 13), (4, 14),
            (5, 12), (5, 13), (5, 14),
        ]
        self.assertCountEqual(expected_indices, output)

    def test_mat_idxs(self):
        idxs = []
        idx = (1, 2)
        n_cosets = 5
        block_size = 3
        in_cols = n_cosets * block_size
        out_cols = (n_cosets * block_size) ** 2
        output = [convert_idx(idx, in_cols, out_cols) for idx in block_indices(idx, block_size)]
        expected = [
            (0, 51), (0, 52), (0, 53),
            (0, 66), (0, 67), (0, 68),
            (0, 81), (0, 82), (0, 83),
        ]
        self.assertCountEqual(expected, output)

    def test_cmm(self):
        th = lambda m: torch.from_numpy(m)
        X = np.random.random((10, 10))
        Y = np.random.random((10, 10))
        Z = make_complex(X, Y)
        A = np.random.random((10, 10))
        B = np.random.random((10, 10))
        C = make_complex(A, B)

        np_res = np.matmul(Z, C)
        real, imag = cmm(th(X), th(Y), th(A), th(B))
        th_res_np = make_complex(real.numpy(), imag.numpy())
        self.assertTrue(np.allclose(np_res, th_res_np))

if __name__ == '__main__':
    unittest.main()
