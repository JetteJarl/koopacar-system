import unittest
from unittest.mock import patch
import rclpy
import os
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#from src.camera_turtlebot.camera_turtlebot.camera_node import Camera
sys.path.append(os.path.dirname(__file__) + "/..camera_task")
from camera_node import Camera

class CameraNodeCaptureImageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # create node to test with camerainput testimg.jpg
        with patch("cv2.VideoCapture", return_value=cv2.VideoCapture(os.path.dirname(__file__) + "/testimg.jpg")):
            self.test_node = Camera()
        self.input_img = cv2.imread(os.path.dirname(__file__) + "/testimg.jpg")

    def tearDown(self):
        self.test_node.destroy_node()

    def test_captureImage(self):
        # read decoy input
        ret, frame = self.test_node.vid.read()
        # check ret, frame for expected values
        self.assertEqual(ret, True)
        self.assertEqual(frame.all(), self.input_img.all())

class CameraNodePublishImageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # cerate node to test
        with patch("cv2.VideoCapture", return_value=cv2.VideoCapture(os.path.dirname(__file__) + "/testimg.jpg")):
            self.test_node = Camera()
        # create decoy node to subscribe to topic
        self.sub_node = rclpy.create_node("sub_camera")
        self.captured_imgmsg = []
        self.sub_node.create_subscription(Image, '/camera_turtlebot/image_raw', lambda img: self.captured_imgmsg.append(img), 10)
        self.input_img = cv2.imread(os.path.dirname(__file__) + "/testimg.jpg")
        self.bridge = CvBridge()

    def tearDown(self):
        self.test_node.destroy_node()
        self.sub_node.destroy_node()

    def test_sendImage(self):
        # spin both nodes once to publish/recive
        rclpy.spin_once(self.test_node)
        rclpy.spin_once(self.sub_node)
        # check recived message
        self.assertEqual(len(self.captured_imgmsg), 1)
        captured_img = self.bridge.imgmsg_to_cv2(self.captured_imgmsg[0])
        self.assertEqual(captured_img.all(), self.input_img.all())

if __name__ == '__main__':
    unittest.main()
