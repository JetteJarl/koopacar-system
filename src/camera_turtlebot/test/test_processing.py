import unittest
import os
import sys
import rclpy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

sys.path.append(os.path.dirname(__file__) + "/../camera_turtlebot")
from image_processing_node import ImgProcsessingNode

class ProcessingNodeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # create node to test
        self.tested_node = ImgProcsessingNode()
        # create decoy node to publish to topic /camera_turtlebot/image_raw
        self.pub_node = rclpy.create_node("pub_image")
        self.publisher = self.pub_node.create_publisher(Image, '/camera_turtlebot/image_raw', 10)
        self.publish_img = cv2.imread(os.path.dirname(__file__) + "/testimg.jpg")
        # create decoy subscriber to subscribe to topic /proc_img
        self.sub_node = rclpy.create_node("sub_compressedImg")
        self.output_img = []
        self.subscriber = self.sub_node.create_subscription(CompressedImage, '/proc_img', lambda msg: self.output_img.append(msg), 10)
        # create cv bridge to convert images
        self.bridge = CvBridge()

    def tearDown(self):
        self.tested_node.destroy_node()
        self.pub_node.destroy_node()
        self.sub_node.destroy_node()

    def test_receiveProcessPublish(self):
        # publish/receive image
        self.publisher.publish(self.bridge.cv2_to_imgmsg(self.publish_img))
        rclpy.spin_once(self.tested_node)
        # receive compressed image
        rclpy.spin_once(self.sub_node)
        # check received message
        self.assertEqual(len(self.output_img), 1)
        self.assertEqual(self.output_img[0], self.bridge.cv2_to_compressed_imgmsg(self.publish_img))

if __name__ == '__main__':
    unittest.main()
