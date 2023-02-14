import unittest
import numpy as np
from src.utils.lidar_cnn import lidar_matrix
import matplotlib.pyplot as plt

from src.utils.file_operations import *


class LidarCNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.test_file_path = "files/test_scan.bin"

    def tearDown(self):
        pass

    @unitest.skip("Not yet implemented. Assertion is missing.")
    def test_lidar_matrix(self):
        # TODO: Define expected test result and add assertion accordingly
        test_input = np.array(list_from_file(self.test_file_path))

        result = lidar_matrix(test_input)




if __name__ == '__main__':
    unittest.main()
