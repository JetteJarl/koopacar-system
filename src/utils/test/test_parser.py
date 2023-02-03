import unittest
import numpy as np

from nav_msgs.msg import Odometry

from src.utils.ros2_message_parser import odom2string
from src.utils.ros2_message_parser import string2odom


class ParserTest(unittest.TestCase):
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

    def test_odom_parsing(self):
        original_odom_msg = Odometry()

        original_odom_msg.header.stamp.sec = 348
        original_odom_msg.header.stamp.nanosec = 990000000
        original_odom_msg.header.frame_id = "odom"

        original_odom_msg.child_frame_id = "base_footprint"

        original_odom_msg.pose.pose.position.x = 1.0
        original_odom_msg.pose.pose.position.y = 1.0
        original_odom_msg.pose.pose.position.z = 1.0

        original_odom_msg.pose.pose.orientation.x = 1.0
        original_odom_msg.pose.pose.orientation.y = 1.0
        original_odom_msg.pose.pose.orientation.z = 1.0
        original_odom_msg.pose.pose.orientation.w = 1.0

        original_odom_msg.pose.covariance = np.zeros(36).tolist()

        original_odom_msg.twist.twist.linear.x = 1.0
        original_odom_msg.twist.twist.linear.y = 1.0
        original_odom_msg.twist.twist.linear.z = 1.0

        original_odom_msg.twist.twist.angular.x = 1.0
        original_odom_msg.twist.twist.angular.y = 1.0
        original_odom_msg.twist.twist.angular.z = 1.0

        original_odom_msg.twist.covariance = np.zeros(36).tolist()

        odom_string = odom2string(original_odom_msg)
        result_odom_msg = string2odom(odom_string)

        self.assertEqual(result_odom_msg, original_odom_msg)


if __name__ == '__main__':
    unittest.main()
