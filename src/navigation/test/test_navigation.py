import unittest
import rclpy
# from src.navigation.navigation import Navigation


class NavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # self.test_navigation_node = Navigation()
        pass

    def tearDown(self):
        # self.test_navigation_node.destroy_node()
        pass


if __name__ == '__main__':
    unittest.main()
