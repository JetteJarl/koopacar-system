import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry

import time
import numpy as np
import os

from src.utils.point_transformation import lidar_data_to_point
from src.utils.point_transformation import remove_inf_ranges
from src.utils.point_transformation import remove_inf_point
from src.utils.point_transformation import inf_ranges_to_zero
from src.utils.ros2_message_parser import odom2string


def combine_seconds_and_nanoseconds(seconds, nanoseconds):
    return seconds + nanoseconds / 1000000000

class LidarDataCollectionNode(Node):
    """Collects and saves Lidar data"""

    def __init__(self):
        super().__init__('lidar_collection')
        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)
        self.subscriber_odom = self.create_subscription(Odometry, 'odom', self.receive_odom, 10)
        self.subscriber_img = self.create_subscription(CompressedImage, '/proc_img', self.receive_img, 10)

        # Constants
        self.SYNC_DEVIATION_ODOM = 0.02  # in seconds
        self.SYNC_DEVIATION_IMG = 0.03 # in seconds
        self.KOOPACAR_HEIGHT = 0.187  # in [m]

        self.LIDAR_PATH = "../../../data/lidar_perception/new_data_set/lidar_scan/"
        self.ODOM_PATH = "../../../data/lidar_perception/new_data_set/odom/"
        self.IMG_PATH = "../../../data/lidar_perception/new_data_set/img/"

        # temp data storage
        self.last_scan = None
        self.recent_odom = []
        self.recent_img = []

        # ros parameters
        self.save_freq = 1  # freq between saves in [s]
        self.store_data = True
        self.declare_parameter("save_freq", value=int(1 / self.save_freq))
        #self.declare_parameter("data_path", value=self.data_path)
        self.declare_parameter("store_data", value=self.store_data)
        self.add_on_set_parameters_callback(self.on_param_change)

        # data collection timer
        self.timer = self.create_timer(self.save_freq, self.timer_callback)

    def received_scan(self, scan):
        """Temporary stores last received scan"""
        self.last_scan = scan

    def receive_odom(self, odom):
        """Store recent odometry messages in list"""
        self.recent_odom.append(odom)

    def receive_img(self, img):
        """Store recent images from processing node"""
        self.recent_img.append(img)

    def timer_callback(self):
        """Saves last stored scan as [x, y ,z] points"""

        if not self.store_data or self.last_scan is None or len(self.recent_img) == 0 or len(self.recent_odom) == 0:
            print("There is data missing that is needed for a snapshot. Check if all necessary topics are active.")
            return

        bridge = CvBridge()

        # retrieve last scan
        ranges = self.last_scan.ranges
        ranges = inf_ranges_to_zero(ranges)
        # points2d = lidar_data_to_point(ranges)
        # points2d = remove_inf_point(points2d)

        # convert to [x, y, z]
        # points3d = np.pad(points2d, ((0, 0), (0, 1)), mode='constant', constant_values=self.KOOPACAR_HEIGHT)

        # synchronize with other data
        stamp_scan_in_seconds = combine_seconds_and_nanoseconds(self.last_scan.header.stamp.sec,
                                                                self.last_scan.header.stamp.nanosec)

        # find matching odom message
        odom = None

        for index, odom_msg in enumerate(self.recent_odom):

            stamp_odom_in_seconds = combine_seconds_and_nanoseconds(odom_msg.header.stamp.sec,
                                                                    odom_msg.header.stamp.nanosec)
            self.recent_odom.pop(index)

            if stamp_odom_in_seconds - stamp_scan_in_seconds < self.SYNC_DEVIATION_ODOM:
                odom = odom_msg
                break

        # find matching img message
        img = None

        for index, img_msg in enumerate(self.recent_img):
            stamp_img_in_seconds = combine_seconds_and_nanoseconds(img_msg.header.stamp.sec,
                                                                   img_msg.header.stamp.nanosec)
            self.recent_img.pop(index)

            if stamp_img_in_seconds - stamp_scan_in_seconds < self.SYNC_DEVIATION_IMG:
                img = bridge.compressed_imgmsg_to_cv2(img_msg)

        if odom is None or img is None:
            raise Exception("Odometry or Image data could not be synced to lidar scan.")

        # generate filenames with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        filename_lidar_scan = os.path.join(self.LIDAR_PATH, "lidar_scan_" + timestamp + ".txt")
        filename_odom = os.path.join(self.ODOM_PATH, "odom_" + timestamp + ".txt")
        filename_img = os.path.join(self.IMG_PATH, "image_" + timestamp + ".jpg")

        # save to file
        file_lidar_scan = open(filename_lidar_scan, mode='w')
        for data in ranges:
            #file_lidar_scan.write(np.array2string(point, precision=3, suppress_small=True, separator=','))
            file_lidar_scan.write(str(data))
            file_lidar_scan.write("\n")
        file_lidar_scan.close()

        file_odom = open(filename_odom, mode="w")
        file_odom.write(odom2string(odom))
        file_odom.close()

        cv2.imwrite(filename_img, img)

        print("Created and saved snapshot.")

    def on_param_change(self, parameters):
        """Changes ros parameters based on parameters."""
        for parameter in parameters:

            if parameter.name == "save_freq":
                save_freq = parameter.value
                print(f"changed save_freq from {self.save_freq} to {save_freq}")
                self.save_freq = save_freq
                # reinitialize timer
                self.timer.destroy()
                self.timer = self.create_timer(self.save_freq, self.timer_callback)
                return SetParametersResult(successful=True)

            #elif parameter.name == "data_path":
            #    data_path = parameter.value
            #    print(f"changed data_path from {self.data_path} to {data_path}")
            #    self.data_path = data_path
            #    return SetParametersResult(successful=True)

            elif parameter.name == "store_data":
                store_data = parameter.value
                print(f"changed store_data from {self.store_data} to {store_data}")
                self.store_data = store_data
                return SetParametersResult(successful=True)

            else:
                print(f"unknown/unused parameter {parameter.name}")
                return SetParametersResult(successful=False)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarDataCollectionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
