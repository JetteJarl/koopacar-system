import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from src.utils.flex_queue import FlexibleQueue


class SensorFusionNode(Node):
    """Receives bounding boxes, cone points, and odometry and publishes cone-/robot-pos."""

    def __init__(self):
        super().__init__('sensor_fusion')

        # create subscriber
        self.subscriber_bboxes_ = self.create_subscription(Float32MultiArray, '/bounding_boxes', self.received_bbox, 10)
        self.subscriber_cone_points = self.create_subscription(Float32MultiArray, '/all_objects', self.received_cone_points, 10)
        # create publisher
        self.publisher_cones = self.create_publisher(Float32MultiArray, '/cone_position', 10)

        # queues for storing messages
        self.bounding_boxes_buffer = FlexibleQueue(10)
        self.cone_points_buffer = FlexibleQueue(10)

    def received_bbox(self, bboxes):
        self.bounding_boxes_buffer.push(bboxes)
        # TODO: Check for matching timestamps on elements in buffers

    def received_cone_points(self, points):
        self.cone_points_buffer.push(points)
        # TODO: Check for matching timestamps on elements in buffers

    def fusion(self, bboxes, cone_points):
        cones = Float32MultiArray()
        # TODO: Implement sensor fusion
        self.publisher_cones.publish(cones)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SensorFusionNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
