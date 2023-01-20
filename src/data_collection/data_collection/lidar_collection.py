import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan


class LidarDataCollectionNode(Node):
    """Collects and saves Lidar data"""
    def __int__(self):
        super().__init__('lidar_collection')
        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)

        # temp data storage
        self.temp_data = None

        # ros parameters
        self.save_freq = 1  # freq between saves in [s]
        self.data_path = "data"
        self.store_data = True
        self.declare_parameter("save_freq", value=int(1 / self.save_freq))
        self.declare_parameter("data_path", value=self.data_path)
        self.declare_parameter("store_data", value=self.store_data)
        self.add_on_set_parameters_callback(self.on_param_change)

        # data collection timer
        self.timer = self.create_timer(self.save_freq, self.timer_callback)

    def received_scan(self, scan):
        """Temporary stores last received scan"""
        self.temp_data = scan

    def timer_callback(self):
        """Saves last stored scan"""
        pass

    def on_param_change(self, parameters):
        """Changes ros parameters based on parameters."""
        for parameter in parameters:

            if parameter.name == "save_freq":
                save_freq = parameter.value
                print(f"changed save_freq from {self.save_freq} to {save_freq}")
                self.save_freq = save_freq
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
