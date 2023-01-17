import rclpy
import cv2
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
import time


class Camera(Node):
    """Captures video stream and publishes images."""

    def __init__(self):
        super().__init__('camera')
        # publisher for raw img data
        self.publisher_ = self.create_publisher(Image, '/camera_turtlebot/image_raw', 10)
        # camera stream
        self.vid = cv2.VideoCapture(0)

        # ros parameters
        self.freq = .2  # freq between images to publish [s]
        self.declare_parameter("fps", int(1 / self.freq))
        self.add_on_set_parameters_callback(self.on_param_change)
        # callback timer
        self.timer = self.create_timer(self.freq, self.callback)

    def callback(self):
        """Captures one frame und publishes to /camera_turtlebot/image_raw."""
        ret, frame = self.vid.read()
        bridge = CvBridge()

        try:
            msg = bridge.cv2_to_imgmsg(frame)
            curr_time = time.time()
            msg.header.stamp.sec = int(curr_time)
            msg.header.stamp.nanosec = int(str(curr_time - int(curr_time)).split('.')[1][:9])
            self.publisher_.publish(msg)
            print(f"send new img {str(datetime.now()).split('.')[0]}")

        except CvBridgeError as e:
            print(e)

    def on_param_change(self, parameters):
        """Changes ros parameters based on parameters."""
        print("--------------------- PARAM CHANGE ---------------------")
        for parameter in parameters:
            if parameter.name == "fps":
                # calc new frequency
                fps = parameter.value
                print(f"changed fps from {1 / self.freq} to {fps}")
                self.freq = 1 / fps
                # reinitialize timer
                tmp = self.timer
                self.timer = self.create_timer(self.freq, self.callback)
                tmp.destroy()
                print("recreated timer")
                return SetParametersResult(successful=True)

        return SetParametersResult(successful=False)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Camera())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
