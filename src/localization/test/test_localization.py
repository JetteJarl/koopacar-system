import unittest
import rclpy
import sys
import os
import numpy as np

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
        self.assertEqual(self.tested_queue.q[0].tolist(), input_elem)

    def test_addOverflowElement(self):
        # create input/output
        input_elem = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        overflow_elem = [5, 5]
        output_elem = [[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
        # add input to queue
        [self.tested_queue.push(elem) for elem in input_elem]
        self.tested_queue.push(overflow_elem)
        # test queue
        self.assertEqual(len(self.tested_queue.q), self.MAX_Q_LEN)
        self.assertEqual(self.tested_queue.q.tolist(), output_elem)

class LocalizationNodeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # create node to test
        self.tested_node = LocalizationNode()

    def tearDown(self):
        pass

    def test_lidarDataToPointCloudZeros(self):
        input_data = np.zeros(360)
        output_data = np.full((360, 2), (0., 0.))
        self.assertAlmostEqual(self.tested_node.lidar_data_to_point_cloud(input_data).tolist(), output_data.tolist())

if __name__ == '__main__':
    unittest.main()
