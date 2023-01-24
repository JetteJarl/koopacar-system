import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
import os
import numpy as np


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
        self.data_path = "lidar_data"
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
        pointsTwoD = self.lidar_data_to_point_cloud(ranges)

        # convert to [x, y, z]
        pointsThreeD = np.pad(pointsTwoD, ((0, 0), (0, 1)), mode='constant', constant_values=self.KOOPACAR_HEIGHT)
        #pointsThreeD = np.ones((len(ranges), 3))
        #for point in pointsTwoD:
        #    np.append(pointsThreeD, np.append(point, self.KOOPACAR_HEIGHT))

        # save to file
        f = open(self.data_path, mode='a')
        for point in pointsThreeD:
            f.write(np.array_str(point, suppress_small=True))
        f.write("\n")
        f.close()

    def lidar_data_to_point_cloud(self, ranges):
        """
        Converts ranges into coordinates.

        Ranges are indexed by angle, and describe the distance until the lidar hit an object.
        Points are returned in array as coordinates in format [x, y].
        """
        number_points = len(ranges)
        points_x = np.array(ranges) * np.sin(np.flip(np.linspace(0, 2 * np.pi, number_points, endpoint=False)))
        points_y = np.array(ranges) * np.cos(np.flip(np.linspace(0, 2 * np.pi, number_points, endpoint=False)))
        points = np.array([[x, y] for x, y in zip(points_x, points_y)])

        return points

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
