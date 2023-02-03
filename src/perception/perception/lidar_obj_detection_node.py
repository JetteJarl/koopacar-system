import os
import rclpy
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, CompressedImage, LaserScan


class LidarObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')

        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)

        # publisher for object positions
        self.publish_pos = self.create_publisher(Float32MultiArray, '/all_objects', 10)

    def received_scan(self, scan):
        print(scan)

    def process_scan(self):
        pass

    def publish_position(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarObjectDetectionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
