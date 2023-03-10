import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from src.utils.flex_queue import FlexibleQueue
from src.utils.message_operations import bbox_msg_to_values, centroid_msg_to_values
#from koopacar_interfaces import Centroids
import numpy as np
import math


class SensorFusionNode(Node):
    """Receives bounding boxes, cone points, and odometry and publishes cone-/robot-pos."""

    def __init__(self):
        super().__init__('sensor_fusion')

        # create subscriber
        self.subscriber_bboxes_ = self.create_subscription(Float32MultiArray, '/bounding_boxes', self.received_bbox, 10)
        self.subscriber_cone_points = self.create_subscription(Float32MultiArray, '/cone_centroids',
                                                               self.received_cone_points, 10)
        # create publisher
        self.publisher_cones = self.create_publisher(Float32MultiArray, '/cone_position', 10)

        # queues for storing messages
        self.bounding_boxes_buffer = FlexibleQueue(30)
        self.cone_points_buffer = FlexibleQueue(30)

        # consts
        self.CAMERA_FOV = 62
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 480
        self.STAMP_DIFF_THRESHOLD = 0.1  # TODO: create as parameter
        self.NUM_COMPARED_SCANS = 5

    def received_bbox(self, msg):
        """
        Safes bboxes to queue and runs sensor fusion

        Label of first dimension: 'Bounding Boxes'
        Size of first dimension: num(bounding boxes) + 1
        Label of second dimension: 'Coordinates'
        Size of second dimension: 6 [top-left-x, top-left-y, bottom-right-x, bottom-right-y, confidence, class]
        Warning: The first element in data consists of the time stamp in format [seconds, nanoseconds, -1, -1, -1, -1]
        """
        self.bounding_boxes_buffer.push(msg)
        if len(self.bounding_boxes_buffer) > 0 and len(self.cone_points_buffer) > 5:
            self.fusion(msg, num_compared_scans=self.NUM_COMPARED_SCANS)

    def received_cone_points(self, msg):
        """
        Safes published message of type Float32MultiArray with centroids to queue.

        Label of first dimension: 'Centroids'
        Size of first dimension: num(centroids) + 1
        Label of second dimension: 'Coordinates'
        Size of second dimension: 2 [x, y]
        Warning: The first element in data consists of the time stamp in format [seconds, nanoseconds]
        """
        self.cone_points_buffer.push(msg)

    def fusion(self, bbox_msg, num_compared_scans=5):
        """
        Returns and publishes the position and color of cones.

        bbox_msg -> bounding box message to match to centroids messages
        """
        # convert message to bounding boxes & timestamp
        bboxes, bboxes_stamp = bbox_msg_to_values(bbox_msg)

        # get synchronized centroid messages
        centroid_msg = self._scans_in_range(bboxes_stamp, self.STAMP_DIFF_THRESHOLD)

        # convert message to centroids
        centroid = centroid_msg_to_values(centroid_msg)[0]

        # match centroids and bounding boxes
        detected_cones = self._detect_cones(bboxes, centroid)

        # create ros2 message
        cones_msg = Float32MultiArray()
        cones_msg.layout.dim.append(MultiArrayDimension())
        cones_msg.layout.dim.append(MultiArrayDimension())
        cones_msg.layout.dim[0].label = 'Detected Cones'
        cones_msg.layout.dim[0].size = len(detected_cones)
        cones_msg.layout.dim[1].label = 'x, y, label'
        cones_msg.layout.dim[1].size = 3

        cones_msg.data = detected_cones.flatten().tolist()

        # publish and return
        self.publisher_cones.publish(cones_msg)
        return cones_msg

    def _scans_in_range(self, stamp, max_diff, return_amount=1):
        """
        Returns list of sequential message from centroid buffer starting with the least diff to stamp.

        If all massages differ more that max_diff nothing is returned.
        stamp         -> timestamp as float in format seconds.nanoseconds
        max_diff      -> max difference as float in format seconds.nanoseconds
        return_amount -> number of returned messages from buffer (must be >= 1)
        """
        # TODO
        #scan_stamps = [self.scan_buffer.get[i] for i in range(len(self.cone_points_buffer))]
        pass

    def _detect_cones(self, bboxes, centroids):
        """
        Returns detected cones from the bounding box and centroids data

        The returned cones is in format [x, y, label].
        """
        if len(centroids) == 0:
            return np.array([])

        # calculate angle of each centroid
        centroid_angles = [math.atan2(-centroid[1], centroid[0]) for centroid in centroids]
        centroid_angles = [math.degrees(centroid_angle) + self.CAMERA_FOV/2 for centroid_angle in centroid_angles]

        # sort centroids by distance to the bot/[0, 0]
        sorted_centroid_indices = np.argsort(np.linalg.norm(centroids, ord=2, axis=1))

        # bool map whether a centroid is assigned to a bounding box
        used_centroids = np.zeros((len(sorted_centroid_indices)))

        # fov to pixels ratio (width)
        fov_px_ratio = self.CAMERA_FOV / self.IMG_WIDTH

        # save labeled centroids
        labeled_centroids = []

        # iterate over bounding boxes, starting with the biggest/highest
        for bb in sorted(bboxes, key=lambda bb: bb[3] - bb[1], reverse=True):
            # approx angles of bounding box start/end
            start_angle = bb[0] * fov_px_ratio
            end_angle = bb[2] * fov_px_ratio

            # check for matching, not used centroids
            for idx in sorted_centroid_indices:
                if start_angle - 1 <= centroid_angles[idx] <= end_angle + 1 and not used_centroids[idx]:
                    labeled_centroids.append((bb[5], centroids[idx]))
                    used_centroids[idx] = 1
                    break

        detected_cones = np.empty((len(labeled_centroids), 3))
        for i, (label, centroid) in enumerate(labeled_centroids):
            detected_cones[i] = centroid[0], centroid[1], label

        return detected_cones


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SensorFusionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
