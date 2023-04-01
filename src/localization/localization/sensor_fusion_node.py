import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from src.utils.flex_queue import FlexibleQueue
from src.utils.message_operations import bbox_msg_to_values, centroid_msg_to_values
from src.utils.message_operations import *
from src.localization.fusion import _detect_cones
#from koopacar_interfaces import Centroids
import numpy as np
import math


# consts
CAMERA_FOV = 62
IMG_WIDTH = 640
IMG_HEIGHT = 480
STAMP_DIFF_THRESHOLD = 10
NUM_COMPARED_SCANS = 5


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
            self.fusion(msg, num_compared_scans=NUM_COMPARED_SCANS)

    def received_cone_points(self, msg):
        """
        Safes published message of type Float32MultiArray with centroids to queue.

        Label of first dimension: 'Centroids'
        Size of first dimension: num(centroids) + 1
        Label of second dimension: 'Coordinates'
        Size of second dimension: 2 [x, y]
        Warning: The first element in data consists of the time stamp in format [seconds, nanoseconds]
        """
        scan = np.array(msg.data).reshape((-1, msg.layout.dim[1].size))
        self.cone_points_buffer.push(scan)

    def fusion(self, bbox_msg, num_compared_scans=5):
        """
        Returns and publishes the position and color of cones.

        bbox_msg -> bounding box message to match to centroids messages
        """
        bboxes, bboxes_stamp = bbox_msg_to_values(bbox_msg)

        centroid_msgs_in_range = self._scans_in_range(bboxes_stamp, STAMP_DIFF_THRESHOLD)

        if len(centroid_msgs_in_range) == 0:
            print("Messages out of sync. Time difference between bboxes and cone centroids is too large.")
            return

        centroid = centroid_msgs_in_range[0][1::]

        # match centroids and bounding boxes
        detected_cones = _detect_cones(bboxes, centroid)

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
        all_scans = np.array([self.cone_points_buffer.get(i) for i in range(len(self.cone_points_buffer))])

        scan_stamps = []
        for scan in all_scans:
            scan_stamp = combine_secs_and_nsecs(scan[0][0], scan[0][1])
            scan_stamps.append(scan_stamp)

        scan_stamps = np.array(scan_stamps)
        time_difference = np.abs(scan_stamps - stamp)

        difference_in_range_indices = np.where(time_difference <= max_diff)
        difference_in_range = time_difference[difference_in_range_indices]
        scans_in_range = all_scans[difference_in_range_indices]

        sorted_difference_indices = np.argsort(difference_in_range)

        return scans_in_range[sorted_difference_indices[:return_amount]]


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SensorFusionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
