import unittest
from unittest.mock import patch, mock_open
import numpy as np
import numpy.testing
from src.utils.parse_from_sdf import bot_pose_from_sdf
from src.utils.parse_from_sdf import cone_position_from_sdf
from src.utils.parse_from_sdf import set_pose_in_sdf


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
        self.test_file.close()

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

        np.testing.assert_allclose(expected_positions_np, res_positions)

    def test_bot_pose_from_sdf(self):
        expected_pose = np.array([-0.143209, -1.16353, 0.00853, 0.00047, 0.005826, 4.3e-05])

        res_pose = bot_pose_from_sdf(self.test_xml_string)

        np.testing.assert_allclose(expected_pose, res_pose)

    def test_set_pose_in_sdf(self):
        with open("files/test_result_set_pose.world") as file_result:
            expected_string = file_result.read()

        pose = [1, 2, -1, -2, -1.5, 1.5]
        name = "KoopaCar"
        file_path = "files/test.world"

        open_mock = mock_open(read_data=self.test_xml_string)
        with patch("builtins.open", open_mock, create=True):
            set_pose_in_sdf(pose, name, file_path)

        open_mock.assert_called_with(file_path, 'w')
        open_mock.return_value.write.assert_called_once_with(expected_string)


if __name__ == '__main__':
    unittest.main()
