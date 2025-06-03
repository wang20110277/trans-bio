import unittest
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.processing import read_fasta, one_hot_encode

class TestDataProcessing(unittest.TestCase):
    
    def test_one_hot_encode(self):
        sequence = "ATCG"
        encoded = one_hot_encode(sequence)
        expected = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # T
            [0, 0, 1, 0],  # C
            [0, 0, 0, 1]   # G
        ])
        np.testing.assert_array_equal(encoded, expected)
        
    def test_one_hot_encode_unknown_nucleotide(self):
        sequence = "ATNX"
        encoded = one_hot_encode(sequence)
        expected = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # T
            [0, 0, 0, 0],  # N (unknown)
            [0, 0, 0, 0]   # X (unknown)
        ])
        np.testing.assert_array_equal(encoded, expected)

if __name__ == '__main__':
    unittest.main()