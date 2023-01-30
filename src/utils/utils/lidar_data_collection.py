import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
import numpy as np
from src.utils.point_transformation import lidar_data_to_point
from src.utils.point_transformation import remove_inf_point
import time


class LidarDataCollectionNode(Node):
    """Collects and saves Lidar data"""

    def __init__(self):
        super().__init__('lidar_collection')
        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)

        # temp data storage
        self.temp_data = None

        # ros parameters
        self.save_freq = 1  # freq between saves in [s]
        self.data_path = "../../../data/raw_lidar_set/"
        self.store_data = True
        self.declare_parameter("save_freq", value=int(1 / self.save_freq))
        self.declare_parameter("data_path", value=self.data_path)
        self.declare_parameter("store_data", value=self.store_data)
        self.add_on_set_parameters_callback(self.on_param_change)

        # data collection timer
        self.timer = self.create_timer(self.save_freq, self.timer_callback)

        # internal variables
        self.KOOPACAR_HEIGHT = 0.187  # in [m]

    def received_scan(self, scan):
        """Temporary stores last received scan"""
        self.temp_data = scan

    def timer_callback(self):
        """Saves last stored scan as [x, y ,z] points"""

        if not self.store_data or self.temp_data is None:
            return

        # retrieve last scan
        scan = self.temp_data
        ranges = scan.ranges
        points2d = lidar_data_to_point(ranges)
        points2d = remove_inf_point(points2d)

        # convert to [x, y, z]
        points3d = np.pad(points2d, ((0, 0), (0, 1)), mode='constant', constant_values=self.KOOPACAR_HEIGHT)
        # points3d = np.ones((len(ranges), 3))
        # for point in points2d:
        #     np.append(points3d, np.append(point, self.KOOPACAR_HEIGHT))

        # generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.data_path + "lidar_scan" + timestamp + ".txt"

        # save to file
        f = open(filename, mode='w')
        for point in points3d:
            f.write(np.array2string(point, precision=3, suppress_small=True, separator=','))
            f.write("\n")
        f.close()

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

            elif parameter.name == "data_path":
                data_path = parameter.value
                print(f"changed data_path from {self.data_path} to {data_path}")
                self.data_path = data_path
                return SetParametersResult(successful=True)

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
