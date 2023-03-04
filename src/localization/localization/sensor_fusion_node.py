import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from src.utils.flex_queue import FlexibleQueue
from src.utils.message_operations import bbox_msg_to_values
#from koopacar_interfaces import Centroids
import numpy as np


class SensorFusionNode(Node):
    """Receives bounding boxes, cone points, and odometry and publishes cone-/robot-pos."""

    def __init__(self):
        super().__init__('sensor_fusion')

        # create subscriber
        self.subscriber_bboxes_ = self.create_subscription(Float32MultiArray, '/bounding_boxes', self.received_bbox, 10)
        self.subscriber_cone_points = self.create_subscription(Float32MultiArray, '/all_objects',
                                                               self.received_cone_points, 10)
        # create publisher
        self.publisher_cones = self.create_publisher(Float32MultiArray, '/cone_position', 10)

        # queues for storing messages
        self.bounding_boxes_buffer = FlexibleQueue(30)
        self.cone_points_buffer = FlexibleQueue(30)

        # consts
        self.CAMERA_FOV = 62
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
            self.fusion(msg)

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

    def fusion(self, bbox_msg):
        """
        Returns and publishes the position and color of cones.

        bbox_msg -> bounding box message to match to centroids messages
        """
        # convert message to bounding boxes & timestamp
        bboxes, bboxes_stamp = bbox_msg_to_values(bbox_msg)

        # get synchronized scan points
        scans = self._scans_in_range(bboxes_stamp, self.STAMP_DIFF_THRESHOLD, self.NUM_COMPARED_SCANS)

        # get fov of buffer


        return bboxes  # TODO: remove return (only for testing during development)
        cones = Float32MultiArray()
        self.publisher_cones.publish(cones)
        return cones

    def _scans_in_range(self, stamp, max_diff, return_amount):
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



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SensorFusionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
