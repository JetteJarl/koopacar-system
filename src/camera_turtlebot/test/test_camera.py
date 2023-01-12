import unittest
from unittest.mock import patch
import rclpy
import os
import cv2

from src.camera_turtlebot.camera_turtlebot.camera_node import Camera



class CameraNodeCaptureImageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        with patch("cv2.VideoCapture", return_value=cv2.VideoCapture(os.path.dirname(__file__) + "/testimg.jpg")):
            self.test_node = Camera()
        self.input_img = cv2.imread(os.path.dirname(__file__) + "/testimg.jpg")

    def tearDown(self):
        self.test_node.destroy_node()

    def test_captureImage(self):
        ret, frame = self.test_node.vid.read()
        self.assertEqual(ret, True)
        self.assertEqual(frame.all(), self.input_img.all())


if __name__ == '__main__':
    unittest.main()
