import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tensorflow as tf

from src.perception.models.lidar.lidar_cnn import create_model
from src.utils.parse_from_sdf import *
from src.utils.plot_data import *


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

        self.cone_label = 1
        self.cone_radius = 0.2

    def received_scan(self, scan):
        model = tf.keras.models.load_model("../../../models/lidar_cnn/")

        ranges = np.array(inf_ranges_to_zero(scan.ranges)).reshape(-1, )
        points = lidar_data_to_point(inf_ranges_to_zero(scan.ranges))
        prediction = model.predict(np.expand_dims(ranges.reshape(1, -1), axis=2)).reshape(-1, )

        labels = np.array([round(pred) for pred in prediction])

        cone_points = points[labels == self.cone_label]
        cone_ranges = ranges.reshape(-1, )[labels == self.cone_label]
        cluster_labels_all = DBSCAN(eps=0.1, min_samples=3).fit_predict(cone_points)

        cone_centers = []

        for index, label in enumerate(np.unique(cluster_labels_all)):
            if label == -1:
                continue

            cluster_points = cone_points[cluster_labels_all == label]
            cluster_ranges = cone_ranges[cluster_labels_all == label]

            closest = cluster_points[np.where(cluster_ranges == np.min(cluster_ranges))][0]

            scalar = self.cone_radius / np.min(cluster_ranges)
            center = closest + scalar * closest

            cone_centers.append(center)

        cone_centers = np.array(cone_centers)

        plot_prediction(cone_centers, points)

        # TODO: Add timestamp
        centroid_msg = Float32MultiArray()
        centroid_msg.layout.dim.append(MultiArrayDimension())
        centroid_msg.layout.dim.append(MultiArrayDimension())
        centroid_msg.layout.dim[0].label = 'Cone Centroids'
        centroid_msg.layout.dim[0].size = cone_centers.shape[0]
        centroid_msg.layout.dim[1].label = 'x, y'
        centroid_msg.layout.dim[1].size = 2

        centroid_msg.data = cone_centers.flatten().tolist()

        self.publish_centroids.publish(centroid_msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LidarObjectDetectionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
