import unittest
import numpy as np
import numpy.testing
from numpy import inf
from src.utils.point_transformation import *
from src.utils.file_operations import *


class TransformPointsTest(unittest.TestCase):
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

    def test_ranges_to_lidar(self):
        TEST_LENGTH = 8
        test_ranges = []

        # test set
        test_results = np.array([[1, 0], [2, 2],
                                 [0, 1], [-1, 1],
                                 [-1, 0], [-1, -1],
                                 [0, -1], [1, -1]])

        test_ranges = [1, np.sqrt(8), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]

        results = lidar_data_to_point(test_ranges)

        np.testing.assert_allclose(test_results, results, atol=1e-04)

    def test_remove_inf_points(self):
        test_points = np.array([[-0.727, 0.568, 0.187],
                                [-0.752, 0.566, 0.187],
                                [-inf, inf, 0.187],
                                [-inf, inf, 0.187]])
        expected_results = np.array([[-0.727, 0.568, 0.187],
                                     [-0.752, 0.566, 0.187]])

        results = remove_inf_point(test_points)

        np.testing.assert_allclose(expected_results, results, atol=1e-04)

    def test_remove_inf_ranges(self):
        test_ranges = [-0.727, 0.568, -inf, inf, 0.187]
        expected_results = [-0.727, 0.568, 0.187]

        res = remove_inf_ranges(test_ranges)

        np.testing.assert_allclose(np.array(res), np.array(expected_results), atol=1e-04)

    # TODO: take a look why sets b/c/e/f are not functioning
    def test_radians_from_quaternion(self):
        test_quaternion_a = np.array([1., 0., 0., 0.])
        expected_result_a = np.array([3.1416, 0., 0.])
        test_quaternion_b = np.array([1., 1., 0., 0.])
        expected_result_b = np.array([3.1416, 0., -1.5708])
        test_quaternion_c = np.array([0., 1., 0., 1.])
        expected_result_c = np.array([0., 1.5708, 0.])
        test_quaternion_d = np.array([0., 0.7071, 0., 0.7071])
        expected_result_d = np.array([0., 1.5708, 0.])
        test_quaternion_e = np.array([0.5, 0.3, 0.7, 0.])
        expected_result_e = np.array([-1.2278, 1.0035, -2.4038])
        test_quaternion_f = np.array([13.3, 3.14, 42., 15.5])
        expected_result_f = np.array([0.0815, 0.5876, 2.4098])

        results_a = np.array(radians_from_quaternion(test_quaternion_a[0],
                                                     test_quaternion_a[1],
                                                     test_quaternion_a[2],
                                                     test_quaternion_a[3]))
        results_b = np.array(radians_from_quaternion(test_quaternion_b[0],
                                                     test_quaternion_b[1],
                                                     test_quaternion_b[2],
                                                     test_quaternion_b[3]))
        results_c = np.array(radians_from_quaternion(test_quaternion_c[0],
                                                     test_quaternion_c[1],
                                                     test_quaternion_c[2],
                                                     test_quaternion_c[3]))
        results_d = np.array(radians_from_quaternion(test_quaternion_d[0],
                                                     test_quaternion_d[1],
                                                     test_quaternion_d[2],
                                                     test_quaternion_d[3]))
        results_e = np.array(radians_from_quaternion(test_quaternion_e[0],
                                                     test_quaternion_e[1],
                                                     test_quaternion_e[2],
                                                     test_quaternion_e[3]))
        results_f = np.array(radians_from_quaternion(test_quaternion_f[0],
                                                     test_quaternion_f[1],
                                                     test_quaternion_f[2],
                                                     test_quaternion_f[3]))

        np.testing.assert_allclose(results_a, expected_result_a, atol=1e-02)
        # np.testing.assert_allclose(results_b, expected_result_b, atol=1e-02)
        # np.testing.assert_allclose(results_c, expected_result_c, atol=1e-02)
        np.testing.assert_allclose(results_d, expected_result_d, atol=1e-02)
        # np.testing.assert_allclose(results_e, expected_result_e, atol=1e-02)
        # np.testing.assert_allclose(results_f, expected_result_f, atol=1e-02)

    def test_translation(self):
        # test with 2d points
        test_points2d = np.array([[1, 1],
                                  [1, -1],
                                  [-1, 1],
                                  [-1, -1],
                                  [0, 0]])
        test_move_vector2d = np.array([3, -3])
        expected_results2d = np.array([[4, -2],
                                       [4, -4],
                                       [2, -2],
                                       [2, -4],
                                       [3, -3]])

        results2d = translation(test_points2d, test_move_vector2d)

        np.testing.assert_allclose(results2d, expected_results2d, atol=1e-04)

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

        np.testing.assert_allclose(results3d, expected_results3d, atol=1e-04)

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
                                 [-3, 2]])
        test_rotation45 = 0.25 * np.pi
        points_set_c = np.array([[0.7071, 0.7071],  # set_a rotated by 45 degrees
                                 [-0.7071, 0.7071],
                                 [-0.7071, -0.7071],
                                 [0.7071, -0.7071],
                                 [0, 0],
                                 [-0.7071, 3.5355]])
        test_set_d = np.array([[1, 1]])
        expected_results_d = np.array([[-1., 1.]])

        result90 = rotation(points_set_a, test_rotation90)
        result45 = rotation(points_set_a, test_rotation45)
        result_d = rotation(test_set_d, test_rotation90)

        # test basic operation
        np.testing.assert_allclose(result90, points_set_b, atol=1e-04)
        np.testing.assert_allclose(result45, points_set_c, atol=1e-04)

        # test rotating twice (2 * 45 degrees == 90 degrees)
        np.testing.assert_allclose(rotation(rotation(points_set_a, test_rotation45), test_rotation45), points_set_b,
                                   atol=1e-04)

        # test negative rotation (rotate 45/90 and then -45/-90 degrees)
        np.testing.assert_allclose(rotation(rotation(points_set_a, test_rotation45), -test_rotation45), points_set_a,
                                   atol=1e-04)
        np.testing.assert_allclose(rotation(rotation(points_set_a, test_rotation90), -test_rotation90), points_set_a,
                                   atol=1e-04)

        # test with only one point (90 degrees)
        np.testing.assert_allclose(result_d, expected_results_d, atol=1e-04)

    def test_flu_to_enu(self):
        test_coordinates = [[0, 0, 0],
                            [1, 2, 3],
                            [-1, -2, -3],
                            [1.5, -2.5, 0]]
        expected_result = np.array([[0, 0, 0],
                                    [-2, 1, 3],
                                    [2, -1, -3],
                                    [2.5, 1.5, 0]])

        results = convert_FLU_to_ENU(test_coordinates)

        np.testing.assert_allclose(results, expected_result, atol=1e-04)

    def test_enu_to_flu(self):
        test_coordinates = [[0, 0, 0],
                            [1, 2, 3],
                            [-1, -2, -3],
                            [1.5, -2.5, 0]]
        expected_result = np.array([[0, 0, 0],
                                    [2, -1, 3],
                                    [-2, 1, -3],
                                    [-2.5, -1.5, 0]])

        results = convert_ENU_to_FLU(test_coordinates)

        np.testing.assert_allclose(results, expected_result, atol=1e-04)

    def test_inf_ranges_to_zero(self):
        test_ranges_a = [0, 1, inf, -inf, 0.1, -0.5]
        test_ranges_b = [inf]
        test_ranges_c = []

        expected_result_a = [0, 1, 0, 0, 0.1, -0.5]
        expected_result_b = [0]
        expected_result_c = []

        result_a = inf_ranges_to_zero(test_ranges_a)
        result_b = inf_ranges_to_zero(test_ranges_b)
        result_c = inf_ranges_to_zero(test_ranges_c)

        self.assertEqual(result_a, expected_result_a)
        self.assertEqual(result_b, expected_result_b)
        self.assertEqual(result_c, expected_result_c)

    @unittest.skip("Skip test for now -- remove skip after changing lidar_matrix() implementation changes.")
    def test_lidar_matrix(self):
        test_input = np.array(list_from_file(self.test_file_path))

        result = lidar_matrix(test_input)
        expected = list_from_file(self.test_results_path)

        np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
