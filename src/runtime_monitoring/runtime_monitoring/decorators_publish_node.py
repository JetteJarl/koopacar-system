import numpy as np
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Float32MultiArray, MultiArrayDimension


class DecoratorPublishNode(Node):
    """
    Publishes a function output.

    Initializing this node will publish a function output of the given
    type to the given topic.
    """

    def __init__(self, data, topic_name):
        super().__init__('decorator_publish_node')

        # create message
        msg = None
        if isinstance(data, int):
            msg = Int32()
            msg.data = data
        elif isinstance(data, float):
            msg = Float32()
            msg.data = data
        elif isinstance(data, str):
            msg = String()
            msg.data = data
        elif isinstance(data, list):
            msg = Float32MultiArray()
            arr = np.array(data)
            for i in range(arr.ndim):
                msg.layout.dim.append(MultiArrayDimension())
            for i, shape in enumerate(arr.shape):
                msg.layout.dim[i].size = shape
            arr = [float(value) for value in arr.flatten().tolist()]
            msg.data = arr
        elif isinstance(data, np.ndarray):
            msg = Float32MultiArray()
            for i in range(data.ndim):
                msg.layout.dim.append(MultiArrayDimension())
            for i, shape in enumerate(data.shape):
                msg.layout.dim[i].size = shape
            arr = [float(value) for value in data.flatten().tolist()]
            msg.data = arr
        else:
            raise ValueError(f"{type(data)} not added to supported message types")

        # create publisher
        self.publisher = self.create_publisher(type(msg), topic_name, 10)

        # publish message
        self.publisher.publish(msg)
