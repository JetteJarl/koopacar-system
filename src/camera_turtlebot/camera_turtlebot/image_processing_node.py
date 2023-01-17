import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from datetime import datetime

class ImgProcsessingNode(Node):
    """Receives image and compresses it to publish it again."""
    def __init__(self):
        super().__init__('img_procsessing_node')
        self.publisher_ = self.create_publisher(CompressedImage, '/proc_img', 10)
        self.subscriber_img_ = self.create_subscription(Image, '/camera_turtlebot/image_raw', self.callback, 10)
        self.bridge = CvBridge()

    def callback(self, msg):
        """Compresses image and publishes it to /proc_img."""
        # forward the image data
        print(f"received new img {str(datetime.now()).split('.')[0]}")

        img = self.bridge.imgmsg_to_cv2(msg)

        # Convert image to compressed image
        compressed_image = self.bridge.cv2_to_compressed_imgmsg(img)
        compressed_image.header.stamp.sec = msg.header.stamp.sec
        compressed_image.header.stamp.nanosec = msg.header.stamp.nanosec
        # Publish compressed image
        self.publisher_.publish(compressed_image)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ImgProcsessingNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
