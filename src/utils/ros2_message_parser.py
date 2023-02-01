from nav_msgs.msg import Odometry
import re
import numpy as np


def odom2string(odom_msg):
    """ Function that parses an Ros2 odometry message to a string. """
    # create header
    seconds = str(odom_msg.header.stamp.sec)
    nanoseconds = str(odom_msg.header.stamp.nanosec)

    header = "header: \n" \
             "  stamp: \n" \
             "    sec: " + seconds + "\n" \
             "    nanosec: " + nanoseconds + "\n" \
             "  frame_id: " + odom_msg.header.frame_id + "\n" \

    # child_frame_id
    child_frame_id = "child_frame_id: " + odom_msg.child_frame_id + " \n"

    # create pose
    position_x = str(odom_msg.pose.pose.position.x)
    position_y = str(odom_msg.pose.pose.position.y)
    position_z = str(odom_msg.pose.pose.position.z)

    orientation_x = str(odom_msg.pose.pose.orientation.x)
    orientation_y = str(odom_msg.pose.pose.orientation.y)
    orientation_z = str(odom_msg.pose.pose.orientation.z)
    orientation_w = str(odom_msg.pose.pose.orientation.w)

    covariance = np.array2string(odom_msg.pose.covariance)

    pose = "pose: \n" \
           "  pose: \n" \
           "    position: \n" \
           "      x: " + position_x + "\n" \
           "      y: " + position_y + "\n" \
           "      z: " + position_z + "\n" \
           "    orientation: \n"  \
           "      x: " + orientation_x + "\n" \
           "      y: " + orientation_y + "\n" \
           "      z: " + orientation_z + "\n" \
           "      w: " + orientation_w + "\n" \
           "  covariance: " + covariance + "\n" \

    # twist
    linear_x = str(odom_msg.twist.twist.linear.x)
    linear_y = str(odom_msg.twist.twist.linear.y)
    linear_z = str(odom_msg.twist.twist.linear.z)

    angular_x = str(odom_msg.twist.twist.angular.x)
    angular_y = str(odom_msg.twist.twist.angular.y)
    angular_z = str(odom_msg.twist.twist.angular.z)

    covariance = np.array2string(odom_msg.twist.covariance)

    twist = "twist: \n" \
            "  twist: \n" \
            "    linear: \n" \
            "      x: " + linear_x + "\n" \
            "      y: " + linear_y + "\n" \
            "      z: " + linear_z + "\n" \
            "  angular: \n" \
            "      x: " + angular_x + "\n" \
            "      y: " + angular_y + "\n" \
            "      z: " + angular_z + "\n" \
            "  covariance: " + covariance + "\n" \

    return header + child_frame_id + pose + twist


def string2odom(odom_string):
    """ This function reverses odom2string() and generates an odometry message from a string"""
    STAMP_SEC = 2
    STAMP_NANOSEC = 3
    STAMP_FRAME = 4

    CHILD_FRAME = 5

    POSE_POS = 9
    POSE_ORIENTATION = 13
    POSE_COV = 17

    TWIST_LIN = 22
    TWIST_ANGULAR = 26
    TWIST_COV = 29

    lines = re.split("\n| \n", odom_string)
    odom_msg = Odometry()

    odom_msg.header.stamp.sec = int(lines[STAMP_SEC].split(": ")[1])
    odom_msg.header.stamp.nanosec = int(lines[STAMP_NANOSEC].split(": ")[1])
    odom_msg.header.frame_id = lines[STAMP_FRAME].split(": ")[1]

    odom_msg.child_frame_id = lines[CHILD_FRAME].split(": ")[1]

    odom_msg.pose.pose.position.x = float(lines[POSE_POS].split(": ")[1])
    odom_msg.pose.pose.position.x = float(lines[POSE_POS + 1].split(": ")[1])
    odom_msg.pose.pose.position.x = float(lines[POSE_POS + 2].split(": ")[1])
    odom_msg.pose.pose.orientation.x = float(lines[POSE_ORIENTATION].split(": ")[1])
    odom_msg.pose.pose.orientation.x = float(lines[POSE_ORIENTATION + 1].split(": ")[1])
    odom_msg.pose.pose.orientation.x = float(lines[POSE_ORIENTATION + 2].split(": ")[1])
    odom_msg.pose.pose.orientation.x = float(lines[POSE_ORIENTATION + 3].split(": ")[1])

    cov = re.split(" ", lines[POSE_COV].split(": ")[1][1:]) + re.split(" ", lines[POSE_COV + 1][1:-1])
    odom_msg.pose.covariance = np.array(cov, dtype=np.float64)

    odom_msg.twist.twist.linear.x = float(lines[TWIST_LIN].split(": ")[1])
    odom_msg.twist.twist.linear.x = float(lines[TWIST_LIN + 1].split(": ")[1])
    odom_msg.twist.twist.linear.x = float(lines[TWIST_LIN + 2].split(": ")[1])
    odom_msg.twist.twist.angular.x = float(lines[TWIST_ANGULAR].split(": ")[1])
    odom_msg.twist.twist.angular.x = float(lines[TWIST_ANGULAR + 1].split(": ")[1])
    odom_msg.twist.twist.angular.x = float(lines[TWIST_ANGULAR + 2].split(": ")[1])

    cov = re.split(" ", lines[TWIST_COV].split(": ")[1][1:]) + re.split(" ", lines[TWIST_COV + 1][1:-1])
    odom_msg.twist.covariance = np.array(cov, dtype=np.float64)

    return odom_msg
