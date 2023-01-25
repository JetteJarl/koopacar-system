import unittest
import numpy as np
from src.utils.point_transformation import lidar_data_to_point


class TransformPointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.TEST_LENGTH = 8
        self.test_ranges = []
        for i in range(0, self.TEST_LENGTH):
            if (i + 1) % 2 == 0:
                self.test_ranges.append(np.sqrt(2))
            else:
                self.test_ranges.append(1)

        self.test_results = np.array([[0, 1], [1, 1],
                                      [1, 0], [1, -1],
                                      [0, -1], [-1, -1],
                                      [-1, 0], [-1, 1]])

    def tearDown(self):
        pass

    def test_transform_point(self):
        results = lidar_data_to_point(self.test_ranges)
        self.assertEqual(results.all(), self.test_results.all())


if __name__ == '__main__':
    unittest.main()
