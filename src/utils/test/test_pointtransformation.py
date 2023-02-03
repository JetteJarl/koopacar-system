import unittest
import numpy as np
from numpy import inf
from src.utils.point_transformation import lidar_data_to_point
from src.utils.point_transformation import remove_inf_point
from src.utils.point_transformation import remove_inf_ranges
from src.utils.point_transformation import translation
from src.utils.point_transformation import rotation


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

    def test_remove_inf_points(self):
        test_points = np.array([[-0.727, 0.568, 0.187],
                                [-0.752, 0.566, 0.187],
                                [-inf,  inf, 0.187],
                                [-inf,  inf, 0.187]])
        expected_results = np.array([[-0.727, 0.568, 0.187],
                                    [-0.752, 0.566, 0.187]])

        results = remove_inf_point(test_points)

        self.assertEqual(expected_results.all(), results.all())

    def test_remove_inf_ranges(self):
        test_ranges = [-0.727, 0.568, -inf,  inf, 0.187]
        expected_results = [-0.727, 0.568, 0.187]

        res = remove_inf_ranges(test_ranges)

        self.assertEqual(np.array(res).all(), np.array(expected_results).all())

    def test_euler_to_radians(self):
        # TODO: Implemented
        pass

    def test_translation(self):
        # test with 2d points
        test_points2d = np.array([[1, 1],
                                  [1, -1],
                                  [-1, 1],
                                  [-1, -1],
                                  [0, 0]])
        test_move_vector2d = np.array([3, -3])
        expected_results2d = np.array([[4, -2],
                                       [3, -4],
                                       [2, 2],
                                       [2, -4],
                                       [3, -3]])

        results2d = translation(test_points2d, test_move_vector2d)

        self.assertEqual(results2d.all(), expected_results2d.all())

        # test with 3d points
        test_points3d = np.array([[1, 1, 1],
                                  [1, 1, -1],
                                  [1, -1, 1],
                                  [1, -1, -1],
                                  [-1, 1, 1],
                                  [-1, 1, -1],
                                  [-1, -1, 1],
                                  [-1, -1, -1],
                                  [0, 0, 0]])
        test_move_vector3d = np.array([3, 0, -3])
        expected_results3d = np.array([[4, 1, -2],
                                       [4, 1, -4],
                                       [4, -1, -2],
                                       [4, -1, -4],
                                       [2, 1, -2],
                                       [2, 1, -4],
                                       [2, -1, -2],
                                       [2, -1, -4],
                                       [3, 0, -3]])

        results3d = translation(test_points3d, test_move_vector3d)

        self.assertEqual(results3d.all(), expected_results3d.all())

    def test_rotation(self):
        points_set_a = np.array([[1, 0],
                               [0, 1],
                               [-1, 0],
                               [0, -1],
                               [0, 0],
                                [2, 3]])
        test_rotation90 = 0.5 * np.pi
        points_set_b = np.array([[0, 1],  # set_a rotated by 90 degrees
                                [-1, 0],
                                [0, -1],
                                [1, 0],
                                [0, 0],
                                [3, -2]])
        test_rotation45 = 0.25 * np.pi
        points_set_c = np.array([[0.7071, 0.7071],  # set_a rotated by 45 degrees
                                 [-0.7071, 0.7071],
                                 [-0.7071, -0.7071],
                                 [0.7071, -0.7071],
                                 [0, 0],
                                 [3.5355, 0.7071]])

        result90 = rotation(points_set_a, test_rotation90)
        result45 = rotation(points_set_a, test_rotation45)

        # test basic operation
        self.assertAlmostEqual(result90.all(), points_set_b.all(), places=4)
        self.assertAlmostEqual(result45.all(), points_set_c.all(), places=4)

        # test rotating twice (2 * 45 degrees == 90 degrees)
        self.assertAlmostEqual(rotation(rotation(points_set_a, test_rotation45), test_rotation45).all(),
                               points_set_c.all())

        # test negative rotation (rotate 45/90 and then -45/-90 degrees)
        self.assertAlmostEqual(rotation(rotation(points_set_a, test_rotation45), -test_rotation45).all(),
                               points_set_a.all())
        self.assertAlmostEqual(rotation(rotation(points_set_a, test_rotation90), -test_rotation90).all(),
                               points_set_a.all())


if __name__ == '__main__':
    unittest.main()
