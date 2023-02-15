import unittest
import numpy as np
from src.utils.file_operations import *


class FileOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.expected_result_01 = np.array([[0., 0., 0.187, 0.],
                                            [0., 0., 0.187, 0.],
                                            [2.06, 0.072, 0.187, 0.],
                                            [1.985, 0.104, 0.187, 0.],
                                            [1.944, 0.136, 0.187, 0.],
                                            [1.937, 0.169, 0.187, 0.],
                                            [1.928, 0.203, 0.187, 0.],
                                            [1.932, 0.237, 0.187, 0.],
                                            [1.939, 0.272, 0.187, 0.],
                                            [1.943, 0.308, 0.187, 0.]])

        self.test_file_path_01 = "files/list01.txt"

        with open(self.test_file_path_01, 'w') as test_file:
            for point in self.expected_result_01:
                test_file.write(np.array2string(point, precision=3, suppress_small=True, separator=','))
                test_file.write("\n")

            test_file.close()

        self.expected_result_02 = np.zeros((256, 256))
        self.test_file_path_02 = "files/list02.txt"

        with open(self.test_file_path_02, 'w') as test_file:
            for point in self.expected_result_02:
                test_file.write(np.array2string(point, precision=3, suppress_small=True, separator=','))
                test_file.write("\n")

            test_file.close()

    def tearDown(self):
        pass

    def test_points_from_file(self):
        result_01 = list_from_file(self.test_file_path_01)
        np.testing.assert_allclose(self.expected_result_01, result_01)

    def test_long_from_file(self):
        result_02 = list_from_file(self.test_file_path_02)
        np.testing.assert_allclose(self.expected_result_02, result_02)

if __name__ == '__main__':
    unittest.main()
