import os
import pickle
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from src.utils.ros2_message_parser import odom2string


class TopicMonitoringNode(Node):
    """Monitores any topic and saves the messages to file."""

    def __init__(self):
        super().__init__('topic_monitoring_node')
        # parameters
        self.message_type = Float32MultiArray
        self.topic_name = '/bounding_boxes'
        self.qos_profile = 10
        #self.declare_parameter("message_type", self.message_type)
        #self.declare_parameter("topic_name", self.topic_name)
        #self.declare_parameter("qos_profile", self.qos_profile)
        #self.add_on_set_parameters_callback(self.on_param_change)

        self.node_starting_time = time.strftime("%Y%m%d-%H%M%S")
        self.data_path = "/home/ubuntu/active-workspace/koopacar-system/monitoring/topics/"
        # subscriber 
        self.subscriber_bboxes_ = self.create_subscription(self.message_type, self.topic_name, self.save_message, self.qos_profile)

    def save_message(self, msg):
        """
        Saves one message from /bounding_boxes and saves it to a file.

        The messages will be stored in the folder described by self.data_path.
        Each time the node starts a new directory is created that stores all
        incoming messages in separate files.
        """
        message_path = os.path.join(self.data_path, self.topic_name.replace("/", ""), self.node_starting_time)
        os.makedirs(message_path, exist_ok=True)
        message_file_path = os.path.join(message_path, self.topic_name.replace("/", "") + time.strftime("%Y%m%d-%H%M%S") + ".log")

        if isinstance(msg, Odometry):
            string_output = odom2string(msg)
            with open(message_file_path, "w") as file:
                file.write(string_output)
        elif isinstance(msg, (Float32MultiArray, LaserScan)):
            with open(message_file_path, "wb") as file:
                pickle.dump(msg, file)

    def on_param_change(self, parameters):
        """Changes ros parameters based on parameters."""
        print("--------------------- PARAM CHANGE ---------------------")
        for parameter in parameters:
            if parameter.name == "message_type":
                message_type = parameter.value
                print(f"Changed message_type from {self.message_type} to {message_type}")
                self.message_type = message_type
                return SetParametersResult(successful=True)
            if parameter.name == "topic_name":
                topic_name = parameter.value
                print(f"Changed topic_name from {self.topic_name} to {topic_name}")
                self.topic_name = topic_name
                return SetParametersResult(successful=True)
            if parameter.name == "qos_profile":
                qos_profile = parameter.value
                print(f"Changed qos_profile from {self.qos_profile} to {qos_profile}")
                self.qos_profile = qos_profile
                return SetParametersResult(successful=True)

        return SetParametersResult(successful=False)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TopicMonitoringNode())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
