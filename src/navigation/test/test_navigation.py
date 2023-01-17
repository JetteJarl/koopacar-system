import unittest
import rclpy
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__) + "/../navigation")
from navigation import Navigation


class NavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_navigation_node = Navigation()

    def tearDown(self):
        self.test_navigation_node.destroy_node()


if __name__ == '__main__':
    unittest.main()
