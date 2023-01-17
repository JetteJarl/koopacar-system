import unittest
import rclpy
import sys
import os
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import math

sys.path.append(os.path.dirname(__file__) + "/../localization")
from localization_node import NpQueue, LocalizationNode


class NpQueueTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.MAX_Q_LEN = 5
        self.ELEM_DIM = 2
        self.tested_queue = NpQueue(self.MAX_Q_LEN, self.ELEM_DIM)

    def tearDown(self):
        pass

    def test_addElement(self):
        # create input
        input_elem = [1, 1]
        # add input to queue
        self.tested_queue.push(input_elem)
        # test queue
        self.assertEqual(input_elem, self.tested_queue.q[0].tolist())

    def test_addElementOverflow(self):
        # create input/output
        input_elem = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        overflow_elem = [5, 5]
        output_elem = [[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
        # add input to queue
        [self.tested_queue.push(elem) for elem in input_elem]
        self.tested_queue.push(overflow_elem)
        # test queue
        self.assertEqual(self.MAX_Q_LEN, len(self.tested_queue.q))
        self.assertEqual(output_elem, self.tested_queue.q.tolist())


# class LocalizationNodeFunctionsTest(unittest.TestCase):
#
#    @classmethod
#    def setUpClass(cls):
#        rclpy.init()
#
#    @classmethod
#    def tearDownClass(cls):
#        rclpy.shutdown()
#
#    def setUp(self):
#        # create node to test
#        self.tested_node = LocalizationNode()
#
#    def tearDown(self):
#        pass
#
#    def test_lidarDataToPointCloudZeros(self):
#        # create input/output
#        input_data = np.zeros(360)
#        output_data = np.full((360, 2), (0., 0.))
#        # test output
#        self.assertAlmostEqual(output_data.tolist(), self.tested_node.lidar_data_to_point_cloud(input_data).tolist())
#
#    def test_lidarDataToPointCloudOnes(self):
#        # create input/output
#        input_data = np.ones(360)
#        output_x = np.array(np.ones(360)) * np.sin(np.flip(np.linspace(0, 2 * np.pi, 360)))
#        output_y = np.array(np.ones(360)) * np.cos(np.flip(np.linspace(0, 2 * np.pi, 360)))
#        output_data = np.array([[x, y] for x, y in zip(output_x, output_y)])
#        # test output
#        self.assertEqual(output_data.tolist(), self.tested_node.lidar_data_to_point_cloud(input_data).tolist())
#
#    def test_removeLidarZeroPoints(self):
#        # create input/output
#       input_data = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [0.1, 0.1]])
#        output_data = [[0, 1], [1, 0], [1, 1], [1e6, 1e6], [1e6, 1e6]]
#        # test output
#        self.assertEqual(output_data, self.tested_node.remove_lidar_zero_points(input_data).tolist())

class ReceiveOdomTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # create node to test
        self.tested_node = LocalizationNode()
        # create decoy node to publish to topic odom
        self.pub_node = rclpy.create_node("pub_odom")
        self.publisher = self.pub_node.create_publisher(Odometry, 'odom', 10)
        # create decoy subscriber to subscribe to topic /robot_pos
        self.sub_node = rclpy.create_node("sub_robotPos")
        self.output_data = []
        self.subscriber = self.sub_node.create_subscription(Float32MultiArray, '/robot_pos',
                                                            lambda msg: self.output_data.append([np.array(msg.data)]),
                                                            10)

    def tearDown(self):
        self.tested_node.destroy_node()
        self.pub_node.destroy_node()
        self.sub_node.destroy_node()

    def test_receiveOdomOnce(self):
        # create input/output
        input_data = Odometry()
        expected_pos = np.array([0, 0])
        expected_orientation = 0

        # publish decoy odom
        self.publisher.publish(input_data)
        # compute odom
        rclpy.spin_once(self.tested_node)
        # receive robot pos
        rclpy.spin_once(self.sub_node)

        # test starting_pos/start_orientation
        self.assertEqual(expected_pos.all(), self.tested_node.start_pos.all())
        self.assertAlmostEqual(expected_orientation, self.tested_node.start_orientation)
        # test pos/orientation
        self.assertEqual(expected_pos.all(), np.array(self.output_data[0]).all())
        self.assertAlmostEqual(expected_orientation, self.tested_node.orientation)

    def test_receiveOdomMultiple(self):
        # create first input with all 0
        input_data_first = Odometry()

        # create second input
        input_data_second = Odometry()
        input_pos_second = np.array([3., -3.])
        input_data_second.pose.pose.position.x = input_pos_second[0]
        input_data_second.pose.pose.position.y = input_pos_second[1]
        expected_start_pos = np.array([0, 0])
        expected_pos = np.array(input_pos_second)

        input_data_second.pose.pose.orientation.z = 1.
        input_data_second.pose.pose.orientation.w = 0.
        expected_start_orientation = 0
        expected_orientation = math.pi

        # publish/compute first decoy odom
        self.publisher.publish(input_data_first)
        rclpy.spin_once(self.tested_node)
        rclpy.spin_once(self.sub_node)
        # publish/compute second decoy odom
        self.publisher.publish(input_data_second)
        rclpy.spin_once(self.tested_node)
        rclpy.spin_once(self.sub_node)

        # test starting_pos/start_orientation
        self.assertEqual(expected_start_pos.all(), self.tested_node.start_pos.all())
        self.assertAlmostEqual(expected_start_orientation, self.tested_node.start_orientation)
        # test pos/orientation
        self.assertEqual(expected_pos.all(), np.array(self.output_data[1]).all())
        self.assertAlmostEqual(expected_orientation, self.tested_node.orientation)


if __name__ == '__main__':
    unittest.main()
