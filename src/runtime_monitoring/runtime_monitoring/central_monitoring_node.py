import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32


class CentralMonitoringNode(Node):

    def __init__(self):
        super().__init__('central_monitoring_node')
        # subscriber params
        self.message_types = [Int32]
        self.topic_names = ['/decorator/time/power']
        self.callbacks = [self.print_message]
        self.qos_profiles = [10]

        # subscriber
        self.subscribers = {}
        for type, name, callback, profile in zip(self.message_types, self.topic_names, self.callbacks, self.qos_profiles):
            self.subscribers[name] = self.create_subscription(type, name, callback, profile)

    def print_message(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        print(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CentralMonitoringNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()