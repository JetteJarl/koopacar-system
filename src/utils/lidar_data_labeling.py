import os
import re
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.utils.plot_data import plot_labled_data_3d
from src.utils.ros2_message_parser import string2odom
from src.utils.point_transformation import euler_from_quaternion
from src.utils.point_transformation import rotation
from src.utils.point_transformation import translation


def list_from_file(file_path):
    try:
        file = open(file_path, "r")

        for line in re.split("\n", file.read()):
            return list(map(float, re.split(",", line[1:-1])))

    except OSError:
        print("Can not open/read the file: " + file_path)


def lidar_labeling_dbscan(data):
    """ Use clustering approach to find groups of points."""

    VAR_UPPER_LIMIT = 1000
    VAR_LOWER_LIMIT = 0.0000001
    CONE_LABEL = 1
    OTHER_LABEL = 0

    EPSILON_RANGE = 0.15
    MIN_SAMPLES = 5

    cluster_labels = DBSCAN(eps=EPSILON_RANGE, min_samples=MIN_SAMPLES).fit_predict(data)

    labels = np.zeros(cluster_labels.shape, dtype=np.uint8)

    for indices, label in enumerate(np.unique(cluster_labels)):
        cluster = data[cluster_labels == label]
        cluster_var = np.var(cluster, axis=0)
        cluster_indices = np.argwhere(cluster_labels == label)

        if VAR_UPPER_LIMIT > np.sum(cluster_var[:2]) > VAR_LOWER_LIMIT and label != -1:
            labels[cluster_indices] = CONE_LABEL
        else:
            labels[indices] = OTHER_LABEL

    return labels


def lidar_labeling_bbox(source_path):
    """ Uses the know position of cones to draw a perimeter for cones matching a cone

        The directory source directory need to have the following structure otherwise
        the necessary files might not be found and an exception will be raised.
        .
        | -- label
                | --
        | -- lidar_scan
                | --
        | -- odom
                | --
        | -- cone_pos.txt
    """

    cone_world_positions = list_from_file(os.path.join(source_path, "cone_pos.txt"))
    # TODO: Calculate initial position relative to bot using bots world pos
    bot_world_position = np.array([-0.143209, -1.16353])

    all_scan_files = sorted(os.listdir(os.path.join(source_path, "lidar_scan")))
    all_odom_files = sorted(os.listdir(os.path.join(source_path, "odom")))

    start_odom = string2odom(all_odom_files[0])
    start_pos = np.array([start_odom.pose.pose.position.x, start_odom.pose.pose.position.y])
    start_orientation_yaw = euler_from_quaternion(start_odom.pose.pose.orientation.x,
                                                  start_odom.pose.pose.orientation.y,
                                                  start_odom.pose.pose.orientation.z,
                                                  start_odom.pose.pose.orientation.w)(2)

    for index, scan_file in enumerate(all_scan_files):
        odom_msg = string2odom(all_odom_files[index])
        current_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        current_orientation_yaw = euler_from_quaternion(odom_msg.pose.pose.orientation.x,
                                                        odom_msg.pose.pose.orientation.y,
                                                        odom_msg.pose.pose.orientation.z,
                                                        odom_msg.pose.pose.orientation.w)[2]

        delta_pos = start_pos - current_pos
        delta_orientation = start_orientation_yaw - current_orientation_yaw

        relative_points = rotation(translation(cone_positions, delta_pos), delta_orientation)

        # TODO: Match scans to area around cone pos
        # -->  height and shape of cones might be relevant
        # -->  bbox or circle area


def main(args=None):
    PATH_TO_SOURCE = "../../data/lidar_perception/new_data_set"
    PATH_TO_DESTINATION = "../../data/00"

    lidar_labeling_bbox(PATH_TO_SOURCE)

    for file in sorted(os.listdir(PATH_TO_SOURCE)):
        try:
            data_input = open(os.path.join(PATH_TO_SOURCE, file), "r")
            data = []
            for line in data_input:
                if line == "\n":
                    continue

                point = list(map(float, line[1:len(line) - 2].split(",")))
                data.append(point)

            labels = lidar_labeling_dbscan(np.array(data))

            label_file = open(os.path.join(PATH_TO_DESTINATION, os.path.splitext(file)[0] + ".label"), 'w')
            for label in labels:
                label_file.write(str(label))
                label_file.write("\n")

            label_file.close()
            data_input.close()

            plot_labled_data_3d(data, labels, title=file)
        except OSError:
            print("Could not open/read file: " + args)


if __name__ == '__main__':
    main()
