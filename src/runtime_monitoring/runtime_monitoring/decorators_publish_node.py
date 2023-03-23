import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Float32MultiArray, MultiArrayDimension


class DecoratorPublishNode(Node):
    """
    Publishes a function output.

    Initializing this node will publish a function output of the given
    type to the given topic.
    """

    def __init__(self, function_output, topic_name):
        super().__init__('decorator_publish_node')

        # create message
        msg = None
        if isinstance(function_output, int):
            msg = Int32()
            msg.data = function_output
        elif isinstance(function_output, float):
            msg = Float32()
            msg.data = function_output
        elif isinstance(function_output, list):
            msg = Float32MultiArray()
            arr = np.array(function_output)
            for i in range(arr.ndim):
                msg.layout.dim.append(MultiArrayDimension())
            for i, shape in enumerate(arr.shape):
                msg.layout.dim[i].size = shape
            arr = [float(value) for value in arr.flatten().tolist()]
            msg.data = arr
        elif isinstance(function_output, np.ndarray):
            msg = Float32MultiArray()
            for i in range(function_output.ndim):
                msg.layout.dim.append(MultiArrayDimension())
            for i, shape in enumerate(function_output.shape):
                msg.layout.dim[i].size = shape
            arr = [float(value) for value in function_output.flatten().tolist()]
            msg.data = arr
        else:
            raise ValueError(f"{type(function_output)} not added to supported message types")

        # create publisher
        self.publisher = self.create_publisher(type(msg), topic_name, 10)

        # publish message
        self.publisher.publish(msg)
