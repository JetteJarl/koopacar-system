import os
import re
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.utils.plot_data import plot_labled_data_3d
from src.utils.ros2_message_parser import string2odom
from src.utils.point_transformation import radians_from_quaternion
from src.utils.point_transformation import rotation
from src.utils.point_transformation import translation
from src.utils.point_transformation import convert_ENU_to_FLU


def list_from_file(file_path):
    try:
        file = open(file_path, "r")

        data = []

        for line in re.split("\n", file.read()):
            if line == '':
                continue

            data.append(list(map(float, re.split(",", line[1:-1]))))

        return data

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


def lidar_labeling(source_path, label_points=True, draw_bboxes=True):
    """ Uses the know position of cones to create bboxes/label the points

        Labels are saves in the following format:
        [label]
        Bounding boxes are saved in following format:
        [class] [center_x] [center_y] [height] [width]
        The directory source directory need to have the following structure otherwise
        the necessary files might not be found and an exception will be raised.
        .
        | -- label
                | --
        | -- lidar_scan
                | --
        | -- odom
                | --
        | -- bboxes
                | --
        | -- cone_pos.txt
    """

    CONE_RADIUS = 0.125  # in [m]
    CONE_HEIGHT = 0.25  # in [m]
    KOOPERCAR_START_POS = [-0.143209, -1.16353]  # [x, y] in [m]
    KOOPERCAR_HEIGHT = 0.187  # in [m]

    # known cone/koopercar position
    cone_world_positions = np.array([np.array(c) for c in list_from_file(os.path.join(source_path, "cone_pos.txt"))])
    bot_world_position = np.array(KOOPERCAR_START_POS)

    # known cone pos relative to koopercar pos
    cone_positions = translation(cone_world_positions[::, 0:2], -bot_world_position)

    # load lidar scan data
    path_to_scan = os.path.join(source_path, "lidar_scan")
    all_scan_files = sorted(os.listdir(path_to_scan))

    # load odometry data
    path_to_odom = os.path.join(source_path, "odom")
    all_odom_files = sorted(os.listdir(path_to_odom))

    # get first odom message, to get start pos
    odom_file = open(os.path.join(path_to_odom, all_odom_files[0]), "r")
    start_odom = string2odom(odom_file.read())
    start_pos = np.array([start_odom.pose.pose.position.x, start_odom.pose.pose.position.y])
    start_orientation_yaw = radians_from_quaternion(start_odom.pose.pose.orientation.x,
                                                    start_odom.pose.pose.orientation.y,
                                                    start_odom.pose.pose.orientation.z,
                                                    start_odom.pose.pose.orientation.w)[2]

    # loop over all lidar scan files, to label
    for index, scan_file in enumerate(all_scan_files):
        # open matching odom file
        odom_file = open(os.path.join(path_to_odom, all_odom_files[index]), "r")
        odom_msg = string2odom(odom_file.read())

        # get pos/orientation at time of lidar scan
        current_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        current_orientation_yaw = radians_from_quaternion(odom_msg.pose.pose.orientation.x,
                                                          odom_msg.pose.pose.orientation.y,
                                                          odom_msg.pose.pose.orientation.z,
                                                          odom_msg.pose.pose.orientation.w)[2]

        # calculate pos/orientation difference since first scan in list
        delta_pos = start_pos - current_pos
        delta_orientation = start_orientation_yaw - current_orientation_yaw

        # calculate pos of known cones relative to koopercar position during lidar scan
        relative_points = rotation(translation(cone_positions, - delta_pos), - delta_orientation)

        # collect points from current scan and converts them to flu-format
        points = np.array([np.array(c) for c in list_from_file(os.path.join(source_path, "lidar_scan", scan_file))])
        points = convert_ENU_to_FLU(points)

        # draw bboxes
        if draw_bboxes:
            _draw_bboxes(source_path, scan_file, relative_points, CONE_RADIUS)

        # label points
        if label_points:
            _label(source_path, scan_file, points, relative_points, CONE_RADIUS)




def _label(source_path, scan_file, points, relative_points, cone_radius):
    # create/open label file
    filename_label = os.path.join(source_path, "label", scan_file.replace(".txt", "") + ".label")
    with open(filename_label, "w") as label_file:
        # loop over all points from scan
        for point in points:
            # loop over all known cones
            cone_hit = False
            for cone in relative_points:
                # check if points are inside cone base area: (x - center_x)² + (y - center_y)² < radius²
                left = pow((point[0] - cone[0]), 2) + pow((point[1] - cone[1]), 2)
                right = pow(cone_radius, 2)
                if left < right:
                    cone_hit = True
                    break

            # write label to file
            if cone_hit:
                label_file.write("1\n")
            else:
                label_file.write("0\n")
def _draw_bboxes(source_path, scan_file, relative_points, cone_radius):
    # create/open label file
    filename_bbox = os.path.join(source_path, "bboxes", scan_file.replace(".txt", "") + ".bbox")
    with open(filename_bbox, "w") as bbox_file:
        # loop over all cones
        for cone in relative_points:
            bbox_file.write(f"1 {cone[0]: .3f} {cone[1]: .3f} {cone_radius:.3f} {cone_radius:.3f}\n")

def main(args=None):
    PATH_TO_SOURCE = "../../data/lidar_perception/new_data_set"
    PATH_TO_DESTINATION = "../../data/00"

    lidar_labeling(PATH_TO_SOURCE)

    # for file in sorted(os.listdir(PATH_TO_SOURCE)):
    #    try:
    #        data_input = open(os.path.join(PATH_TO_SOURCE, file), "r")
    #        data = []
    #        for line in data_input:
    #            if line == "\n":
    #                continue

    #            point = list(map(float, line[1:len(line) - 2].split(",")))
    #            data.append(point)

    #        labels = lidar_labeling_dbscan(np.array(data))

    #        label_file = open(os.path.join(PATH_TO_DESTINATION, os.path.splitext(file)[0] + ".label"), 'w')
    #        for label in labels:
    #            label_file.write(str(label))
    #            label_file.write("\n")

    #        label_file.close()
    #        data_input.close()

    #        plot_labled_data_3d(data, labels, title=file)
    #    except OSError:
    #        print("Could not open/read file: " + args)


if __name__ == '__main__':
    main()
