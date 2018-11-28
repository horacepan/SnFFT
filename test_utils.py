import unittest
from utils import *

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

if __name__ == '__main__':
    unittest.main()
