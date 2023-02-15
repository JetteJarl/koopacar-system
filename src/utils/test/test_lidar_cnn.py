import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.utils.file_operations import *
from src.utils.lidar_cnn import lidar_matrix


class LidarCNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.test_file_path = "files/test_scan.bin"
        self.test_results_path = "files/test_result_256.txt"

    def tearDown(self):
        pass

    # @unittest.skip("Not yet implemented. Assertion is missing.")
    def test_lidar_matrix(self):
        test_input = np.array(list_from_file(self.test_file_path))

        result = lidar_matrix(test_input)
        expected = list_from_file(self.test_results_path)

        np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
