import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from src.utils.flex_queue import FlexibleQueue
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
        self.FOV = 62

    def received_bbox(self, msg):
        """Safes bboxes to queue and runs sensor fusion"""
        self.bounding_boxes_buffer.push(msg)
        if len(self.bounding_boxes_buffer) > 0 and len(self.cone_points_buffer) > 5:
            self.fusion(msg)

    def received_cone_points(self, msg):
        self.cone_points_buffer.push(msg)

    def fusion(self, bbox_msg):
        # save bounding boxes
        bboxes = np.array(bbox_msg.data)
        bboxes = bboxes.reshape((-1, bbox_msg.layout.dim[1].size))[1:]

        # save time stamp of bounding boxes
        bboxes_stamp_sec = int(bboxes[0, 0])
        bboxes_stamp_nano = int(bboxes[0, 1])
        bboxes_stamp = float(f"{bboxes_stamp_sec}.{bboxes_stamp_nano}")

        # get synchronized scan points
        scans = []  # TODO (how is /all_objects msg formatted?)

        # get fov of buffer
        buffer_fov = [np.concatenate((item[int(-self.FOV / 2):], item[:int(self.FOV / 2)])) for item in scans]

        return bboxes  # TODO: remove return (only for testing during development)
        cones = Float32MultiArray()
        self.publisher_cones.publish(cones)
        return cones


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SensorFusionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
