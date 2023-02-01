from nav_msgs.msg import Odometry


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

    covariance = str(odom_msg.pose.covariance)

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

    covariance = str(odom_msg.twist.covariance)

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

    lines = odom_string.split("\n")
    odom_msg = Odometry()

    for line in lines:
        match line:
            case line.starts
