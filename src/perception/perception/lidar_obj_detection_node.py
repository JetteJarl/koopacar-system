import os
import rclpy
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, CompressedImage, LaserScan
import tensorflow as tf
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from src.utils.point_transformation import *
from src.utils.plot_data import *
from src.perception.models.lidar.lidar_cnn import create_model


class LidarObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')

        # subscribe to /scan to receive LiDAR data
        self.subscriber_scan = self.create_subscription(LaserScan, 'scan', self.received_scan,
                                                        qos_profile=qos_profile_sensor_data)

        # publisher for object positions
        self.publish_pos = self.create_publisher(Float32MultiArray, '/all_objects', 10)

        self.cone_label = 1

    def received_scan(self, scan):
        model = create_model()
        model.load_weights("../models/lidar/weights/")

        ranges = np.expand_dims(np.array(inf_ranges_to_zero(scan.ranges)).reshape(1, -1), axis=2)
        points = lidar_data_to_point(inf_ranges_to_zero(scan.ranges))
        prediction = model.predict(ranges).reshape(-1,)

        plot_labled_data_3d(points, prediction)

        # labels = np.array([round(pred) for pred in prediction])
        # cone_points = points[labels == self.cone_label]
        # cluster_labels = DBSCAN(eps=0.1, min_samples=3).fit_predict(cone_points)
        # centroids = []

        # for index, label in enumerate(np.unique(cluster_labels)):
        #     cluster = cone_points[cluster_labels == label]
        #     centroids.append(np.mean(cluster, axis=0))

        # TODO: Publish centroids

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
