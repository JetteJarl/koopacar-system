import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tensorflow as tf

from src.perception.models.lidar.lidar_cnn import *
from src.perception.models.lidar.cones_from_prediction import *
from src.utils.parse_from_sdf import *
from src.utils.plot_data import *


CONE_LABEL = 1
CONE_RADIUS = 0.2


def plot_prediction(cone_centers, points):
    plt.scatter(-cone_centers[:, 1], cone_centers[:, 0], c='red', alpha=0.5, label="predicted cone centers")
    plt.scatter(-points[:, 1], points[:, 0], c='black', s=0.5, label="lidar scan")

    # world_file = open(
    #     "/home/ubuntu/koopacar-simulation-assets/install/koopacar_simulation/share/koopacar_simulation/worlds/track01_circle.world")
    # actual_centers = cone_position_from_sdf(world_file.read())

    # plt.scatter(-actual_centers[:, 1], actual_centers[:, 0], c='green', alpha=0.5, label="actual cone centers")

    plt.legend()
    plt.show()


class LidarObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')

        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)

        # publisher for object positions
        self.publish_centroids = self.create_publisher(Float32MultiArray, '/cone_centroids', 10)

        self.model = tf.keras.models.load_model("../models/lidar/model/")

    def received_scan(self, scan):
        ranges = np.array(inf_ranges_to_zero(scan.ranges)).reshape(-1, )
        points = lidar_data_to_point(inf_ranges_to_zero(scan.ranges))
        prediction = self.model.predict(np.expand_dims(ranges.reshape(1, -1), axis=2))

        labels = probability_to_labels(prediction).reshape(-1,)

        cones = get_cone_centroids(labels, points, ranges)
        cone_centers = np.array([c[0] for c in cones])

        plot_prediction(cone_centers, points)

        # Format timestamp
        time_stamp = np.zeros((1, cone_centers.shape[1]))
        time_stamp[0] = np.array([scan.header.stamp.sec, scan.header.stamp.nanosec], dtype=int)

        centroid_msg = Float32MultiArray()
        centroid_msg.layout.dim.append(MultiArrayDimension())
        centroid_msg.layout.dim.append(MultiArrayDimension())
        centroid_msg.layout.dim[0].label = 'Cone Centroids (first entry is timestamp)'
        centroid_msg.layout.dim[0].size = cone_centers.shape[0]
        centroid_msg.layout.dim[1].label = 'x, y'
        centroid_msg.layout.dim[1].size = 2

        data = np.vstack((time_stamp, cone_centers))
        centroid_msg.data = data.flatten().tolist()

        self.publish_centroids.publish(centroid_msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarObjectDetectionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
