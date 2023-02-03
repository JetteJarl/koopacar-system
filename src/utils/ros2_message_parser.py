from nav_msgs.msg import Odometry
import re
import numpy as np


def odom2string(odom_msg):
    """ Function that parses a Ros2 odometry message to a string. """
    # create header
    seconds = str(odom_msg.header.stamp.sec)
    nanoseconds = str(odom_msg.header.stamp.nanosec)

    header = "header: \n" \
             "  stamp: \n" \
             "    sec: " + seconds + "\n" \
             "    nanosec: " + nanoseconds + "\n" \
             "  frame_id: " + odom_msg.header.frame_id + "\n" \

    # child_frame_id
    child_frame_id = "child_frame_id: " + odom_msg.child_frame_id + "\n"

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
    """ This function reverses odom2string() and generates an odometry message from a string. """

    if not isinstance(odom_string, str):
        raise Exception("Expected string and got " + str(type(odom_string)))

    # Define regular expressions
    stamp_sec_reg = re.compile("\ssec:.*")
    stamp_nanosec_reg = re.compile("\snanosec:.*")
    stamp_frame_reg = re.compile("\sframe_id:.*")

    child_frame_reg = re.compile("child_frame_id:.*")

    cov_reg = re.compile("covariance:.*?]", flags=re.DOTALL)
    x_reg = re.compile("\sx:.*")
    y_reg = re.compile("\sy:.*")
    z_reg = re.compile("\sz:.*")
    w_reg = re.compile("\sw:.*")

    odom_msg = Odometry()

    odom_msg.header.stamp.sec = int(stamp_sec_reg.findall(odom_string)[0].split(": ")[1])
    odom_msg.header.stamp.nanosec = int(stamp_nanosec_reg.findall(odom_string)[0].split(": ")[1])
    odom_msg.header.frame_id = stamp_frame_reg.findall(odom_string)[0].split(": ")[1]

    odom_msg.child_frame_id = child_frame_reg.findall(odom_string)[0].split(": ")[1]

    odom_msg.pose.pose.position.x = float(x_reg.findall(odom_string)[0].split(": ")[1])
    odom_msg.pose.pose.position.y = float(y_reg.findall(odom_string)[0].split(": ")[1])
    odom_msg.pose.pose.position.z = float(z_reg.findall(odom_string)[0].split(": ")[1])
    odom_msg.pose.pose.orientation.x = float(x_reg.findall(odom_string)[1].split(": ")[1])
    odom_msg.pose.pose.orientation.y = float(y_reg.findall(odom_string)[1].split(": ")[1])
    odom_msg.pose.pose.orientation.z = float(z_reg.findall(odom_string)[1].split(": ")[1])
    odom_msg.pose.pose.orientation.w = float(w_reg.findall(odom_string)[0].split(": ")[1])

    odom_msg.twist.twist.linear.x = float(x_reg.findall(odom_string)[2].split(": ")[1])
    odom_msg.twist.twist.linear.y = float(y_reg.findall(odom_string)[2].split(": ")[1])
    odom_msg.twist.twist.linear.z = float(z_reg.findall(odom_string)[2].split(": ")[1])
    odom_msg.twist.twist.angular.x = float(x_reg.findall(odom_string)[3].split(": ")[1])
    odom_msg.twist.twist.angular.y = float(y_reg.findall(odom_string)[3].split(": ")[1])
    odom_msg.twist.twist.angular.z = float(z_reg.findall(odom_string)[3].split(": ")[1])

    cov = cov_reg.findall(odom_string)

    pose_cov_str = re.split("\[|]", cov[0])[1].replace("\n", "")
    pose_cov = np.fromstring(pose_cov_str, dtype=float, sep=' ')

    twist_cov_str = re.split("\[|]", cov[1])[1].replace("\n", "")
    twist_cov = np.fromstring(twist_cov_str, dtype=float, sep=' ')

    odom_msg.pose.covariance = pose_cov
    odom_msg.twist.covariance = twist_cov

    return odom_msg
