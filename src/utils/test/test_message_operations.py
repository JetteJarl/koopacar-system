import unittest
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from src.utils.message_operations import bbox_msg_to_values


class MyTestCase(unittest.TestCase):
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

    def test_bbox_msg_to_values(self):
        test_bbox_a = [1., 1., 50., 50., 95., 1.]
        test_bbox_b = [100., 1., 150., 50., 95., 1.]
        test_timestamp = [130., 4000., -1., -1., -1., -1.]

        test_bbox_msg_a = Float32MultiArray()
        test_bbox_msg_a.layout.dim.append(MultiArrayDimension())
        test_bbox_msg_a.layout.dim.append(MultiArrayDimension())
        test_bbox_msg_a.layout.dim[0].label = 'Bounding Boxes'
        test_bbox_msg_a.layout.dim[0].size = 3  # 1 time stamp + 2 bounding boxes
        test_bbox_msg_a.layout.dim[1].label = 'Coordinates'
        test_bbox_msg_a.layout.dim[1].size = 6  # (top-left-x, top-left-y, bottom-right-x, bottom-right-y, confidence, class)
        test_bbox_msg_a.data = test_timestamp + test_bbox_a + test_bbox_b

        expected_result_data_a = np.array([[1., 1., 50., 50., 95., 1.], [100., 1., 150., 50., 95., 1.]])
        expected_result_timestamp_a = 130.4000

        test_bbox_msg_b = Float32MultiArray()
        test_bbox_msg_b.layout.dim.append(MultiArrayDimension())
        test_bbox_msg_b.layout.dim.append(MultiArrayDimension())
        test_bbox_msg_b.layout.dim[0].label = 'Bounding Boxes'
        test_bbox_msg_b.layout.dim[0].size = 1  # 1 time stamp + 2 bounding boxes
        test_bbox_msg_b.layout.dim[1].label = 'Coordinates'
        test_bbox_msg_b.layout.dim[1].size = 6
        test_bbox_msg_b.data = test_timestamp

        expected_result_data_b = np.array([])
        expected_result_timestamp_b = 130.4000

        result_data_a, result_timestamp_a = bbox_msg_to_values(test_bbox_msg_a)
        result_data_b, result_timestamp_b = bbox_msg_to_values(test_bbox_msg_b)

        np.testing.assert_allclose(expected_result_data_a, result_data_a, atol=1e-04)
        self.assertAlmostEqual(expected_result_timestamp_a, result_timestamp_a)

        np.testing.assert_allclose(expected_result_data_b, result_data_b, atol=1e-04)
        self.assertAlmostEqual(expected_result_timestamp_b, result_timestamp_b)


if __name__ == '__main__':
    unittest.main()
