import unittest
import numpy as np
from numpy import inf
from src.utils.point_transformation import lidar_data_to_point
from src.utils.point_transformation import remove_inf_point


class TransformPointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_transform_point(self):
        TEST_LENGTH = 8
        test_ranges = []

        for i in range(0, TEST_LENGTH):
            if (i + 1) % 2 == 0:
                test_ranges.append(np.sqrt(2))
            else:
                test_ranges.append(1)

        test_results = np.array([[0, 1], [1, 1],
                                [1, 0], [1, -1],
                                [0, -1], [-1, -1],
                                [-1, 0], [-1, 1]])

        results = lidar_data_to_point(test_ranges)
        self.assertEqual(results.all(), test_results.all())

    def test_remove_inf(self):
        test_points = np.array([[-0.727, 0.568, 0.187],
                                [-0.752, 0.566, 0.187],
                                [-inf,  inf, 0.187],
                                [-inf,  inf, 0.187]])
        expected_results = np.array([[-0.727, 0.568, 0.187],
                                    [-0.752, 0.566, 0.187]])

        results = remove_inf_point(test_points)

        self.assertEqual(expected_results.all(), results.all())


if __name__ == '__main__':
    unittest.main()
