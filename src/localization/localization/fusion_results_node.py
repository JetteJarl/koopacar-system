import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class FusionResults(Node):
    def __init__(self):
        super().__init__('fusion_results_node')

        self.cone_subscriber = self.create_subscription(Float32MultiArray, '/cone_position', self.plot_results, 10)

    def plot_results(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FusionResults())

    rclpy.shutdown()


if __name__ == '__main__':
    main()