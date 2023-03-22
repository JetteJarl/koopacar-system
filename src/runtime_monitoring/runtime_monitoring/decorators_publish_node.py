from rclpy.node import Node


class DecoratorPublishNode(Node):
    """
    Publishes a function output.

    Initializing this node will publish a function output of the given
    type to the given topic.
    """

    def __init__(self, function_output, topic_name, message_type):
        super().__init__('decorator_publish_node')

        # publisher
        self.publisher = self.create_publisher(message_type, topic_name, 10)
        msg = message_type()
        msg.data = function_output
        self.publisher.publish(msg)
