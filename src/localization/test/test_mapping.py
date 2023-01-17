import unittest
import rclpy
import sys
import os
import numpy as np
from src.localization.localization.mapping_node import FlexibleQueue, MappingNode
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class FlexibleQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.MAX_Q_LEN = 5
        self.test_queue = FlexibleQueue(self.MAX_Q_LEN)

    def tearDown(self):
        pass

    def test_push(self):
        # Test adding single element
        test_element = [0, 0.0, 0.0]
        self.test_queue.push(test_element)

        self.assertEqual([test_element], self.test_queue.queue)

        # Test adding multiple elements leading to overflow
        test_overflow_list = [[1, 1.1, 1.1], [2, 2.2, 2.2], [3, 3.3, 3.3], [4, 4.4, 4.4], [5, 5.5, 5.5]]
        expected_content = test_overflow_list.copy()
        expected_content.reverse()

        for element in test_overflow_list:
            self.test_queue.push(element)

        self.assertEqual(expected_content, self.test_queue.queue)

    def test_get(self):
        self.assertRaises(IndexError, self.test_queue.get, self.MAX_Q_LEN)


class MappingNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_mapping_node = MappingNode()

    def tearDown(self):
        self.test_mapping_node.destroy_node()

    def test_receive_robot_pos(self):
        # TODO: Change assertion when changing type of self.test_mapping_node.robot_pos
        test_position = [1.0, 2.0]

        # New message with type Float32MultiArray
        pos_msg = Float32MultiArray()

        pos_msg.layout.dim.append(MultiArrayDimension())
        pos_msg.layout.dim.append(MultiArrayDimension())
        pos_msg.layout.dim[0].label = 'Robot position'
        pos_msg.layout.dim[0].size = 1
        pos_msg.layout.dim[1].label = 'x,y'
        pos_msg.layout.dim[1].size = 2

        pos_msg.data = test_position

        # Call receive_robot_pos callback with custom message
        self.test_mapping_node.receive_robot_pos(pos_msg)

        # Compare results
        self.assertEqual(test_position, self.test_mapping_node.robot_pos[0].tolist())

    def test_compare_cones(self):
        test_cone_a = np.array([0, 1.0, 1.0])
        test_cone_b = np.array([0, 1.0 + self.test_mapping_node.TRACKING_RADIUS - 0.01, 1.0])
        test_cone_c = np.array([0, 1.0 + self.test_mapping_node.TRACKING_RADIUS, 1.0 + self.test_mapping_node.TRACKING_RADIUS])
        test_cone_d = np.array([1, 1.0, 1.0])

        # comparison with itself
        self.assertTrue(self.test_mapping_node.compare_cones(test_cone_a, test_cone_a), msg="Comparison with itself.")

        # comparison with cone in range
        self.assertTrue(self.test_mapping_node.compare_cones(test_cone_a, test_cone_b), msg="Comparison with cone in "
                                                                                            "range")

        # comparison with cone out of range
        self.assertFalse(self.test_mapping_node.compare_cones(test_cone_a, test_cone_c), msg="Comparison with cone out "
                                                                                            "of range.")

        # comparison with different label
        self.assertFalse(self.test_mapping_node.compare_cones(test_cone_a, test_cone_d), msg="Comparison with different"
                                                                                            "label.")

    def test_find_cone_neighbor(self):
        test_cone_list = np.array([[0, 1.0, 1.0],
                          [1, 1.0, 1.0],
                          [2, 2.0, 3.0]])

        test_cone_a = np.array([1, 1.0, 1.0])
        test_cone_b = np.array([2, 5.0, 5.0])

        self.assertAlmostEqual(np.array(test_cone_a).all(),
                               np.array(self.test_mapping_node.find_cone_neighbor_in_list(test_cone_a,
                                                                                          test_cone_list)).all(),
                               msg="Find existing cone. Should be match.")

        self.assertIsNone(np.array(self.test_mapping_node.find_cone_neighbor_in_list(test_cone_b,
                                                                                     test_cone_list)).all(),
                          msg="Looking for not existing cone. Should be None.")

    def test_track_cones(self):
        test_data = [
            np.array([[0, 1.0, 1.0],
                      [1, 2.0, 2.0],
                      [2, 6.0, 6.0]]),
            np.array([[0, 0.9, 1.1],
                      [1, 2.0, 2.0],
                      [2, 6.0, 6.0]]),
            np.array([[0, 1.0, 1.0],
                      [1, 2.0, 2.0],
                      [2, 6.0, 6.0],
                      [0, 3.5, 2.0]]),
            np.array([[0, 1.1, 0.9],
                      [1, 2.0, 2.0],
                      [2, 6.0, 6.0]]),
            np.array([[0, 1.0, 1.0],
                      [1, 2.0, 2.0],
                      [2, 6.0, 6.0]])
        ]

        filler_frame = np.array([[0, 1.0, 1.0],
                                 [1, 2.0, 2.0],
                                 [2, 6.0, 6.0]])

        expected_cones = np.array([[0, 1.0, 1.0],
                                   [1, 2.0, 2.0],
                                   [2, 6.0, 6.0]])

        for test_frame in test_data:
            self.test_mapping_node.cone_buffer.push(test_frame)

        while self.test_mapping_node.cone_buffer.size < self.test_mapping_node.BUFFER_LENGTH:
            self.test_mapping_node.cone_buffer.push(filler_frame)

        self.test_mapping_node.track_cones()

        self.assertEqual(expected_cones.all(), np.array(self.test_mapping_node.known_cones).all())

    def test_receive_cones(self):
        pass

    def test_publish_cones(self):
        pass


if __name__ == '__main__':
    unittest.main()
