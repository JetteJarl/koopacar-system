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

    def test_fusion(self):
        test_bbox_a = [1., 1., 50., 50., 95., 1.]
        test_bbox_b = [100., 1., 150., 50., 95., 1.]

        test_msg = Float32MultiArray()
        test_msg.layout.dim.append(MultiArrayDimension())
        test_msg.layout.dim.append(MultiArrayDimension())
        test_msg.layout.dim[0].label = 'Bounding Boxes'
        test_msg.layout.dim[0].size = 2  # 2 bounding boxes
        test_msg.layout.dim[1].label = 'Coordinates'
        test_msg.layout.dim[1].size = 6  # (top-left-x, top-left-y, bottom-right-x, bottom-right-y, confidence, class)
        test_msg.data = test_bbox_a + test_bbox_b

        expected_result = np.array([1., 1., 50., 50., 95., 1., 100., 1., 150., 50., 95., 1.])

        result = self.test_sensor_fusion_node.fusion(test_msg)

        np.testing.assert_allclose(result, expected_result)

if __name__ == '__main__':
    unittest.main()
