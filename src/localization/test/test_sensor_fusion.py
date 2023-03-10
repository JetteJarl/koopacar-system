import unittest
import rclpy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
from src.localization.localization.sensor_fusion_node import SensorFusionNode


class SensorFusionNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_sensor_fusion_node = SensorFusionNode()

    def tearDown(self):
        self.test_sensor_fusion_node.destroy_node()

    @unittest.skip
    def test_fusion(self):
        # TODO
        test_bbox_a = [1., 1., 50., 50., 95., 1.]
        test_bbox_b = [100., 1., 150., 50., 95., 1.]

        test_bbox_msg = Float32MultiArray()
        test_bbox_msg.layout.dim.append(MultiArrayDimension())
        test_bbox_msg.layout.dim.append(MultiArrayDimension())
        test_bbox_msg.layout.dim[0].label = 'Bounding Boxes'
        test_bbox_msg.layout.dim[0].size = 2  # 2 bounding boxes
        test_bbox_msg.layout.dim[1].label = 'Coordinates'
        test_bbox_msg.layout.dim[1].size = 6  # (top-left-x, top-left-y, bottom-right-x, bottom-right-y, confidence, class)
        test_bbox_msg.data = test_bbox_a + test_bbox_b

        expected_result = np.array([1., 1., 50., 50., 95., 1., 100., 1., 150., 50., 95., 1.])

        result = self.test_sensor_fusion_node.fusion(test_bbox_msg)

        np.testing.assert_allclose(result, expected_result)

    @unittest.skip
    def test_scans_in_range(self):
        # TODO
        pass

    def test_detect_cones(self):
        # TODO: expand
        test_bboxes_a = np.array([[300., 10., 340., 60., 95., 1.], [600., 10., 640., 60., 95., 0.]])
        test_centroids_a = np.array([[1., 0.], [1.664, -1.]])

        expected_result_a = np.array([[1., 0., 1.], [1.664, -1., 0.]])

        result_a = self.test_sensor_fusion_node._detect_cones(test_bboxes_a, test_centroids_a)

        np.testing.assert_allclose(result_a, expected_result_a)


if __name__ == '__main__':
    unittest.main()
