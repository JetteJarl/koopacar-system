import unittest
import numpy as np
import numpy.testing
from src.utils.parse_from_sdf import bot_pose_from_sdf
from src.utils.parse_from_sdf import cone_position_from_sdf


class ParserSDFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.test_file = open("files/test.world")
        self.test_xml_string = self.test_file.read()

    def tearDown(self):
        pass

    def test_cone_pos_from_sdf(self):
        expected_positions = [[1.31478, -0.412202, -0.0],
                              [2.14463, -1.14706, -0.0],
                              [0.548962, -1.97284, -0.0],

                              [2.0559, 0.2345, -0.0],
                              [1.32091, -1.39194, -0.0],
                              [0.587561, 0.723581, -0.0],
                              [2.44373, -3.00585, -0.0],

                              [0.568345, -0.451634, 0.0],
                              [2.64047, -0.57498, -0.0],
                              [1.8662, -1.83572, -0.0]]

        expected_positions_np = np.array([np.array(c, dtype=float) for c in expected_positions])

        res_positions = cone_position_from_sdf(self.test_xml_string)

        np.testing.assert_almost_equal(expected_positions_np, res_positions)

    def test_bot_pose_from_sdf(self):
        expected_pose = ([], )

        res_pose = bot_pose_from_sdf(self.test_xml_string)

        self.assertEqual(expected_pose[1], res_pose[1])
        np.testing.assert_allclose(expected_pose)


if __name__ == '__main__':
    unittest.main()
