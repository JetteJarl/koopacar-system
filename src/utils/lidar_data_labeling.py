import os
from sklearn.cluster import DBSCAN
import numpy as np
import re

from src.utils.ros2_message_parser import string2odom
from src.utils.point_transformation import radians_from_quaternion
from src.utils.point_transformation import rotation
from src.utils.point_transformation import translation
from src.utils.point_transformation import lidar_data_to_point
from src.utils.parse_from_sdf import bot_pose_from_sdf
from src.utils.parse_from_sdf import cone_position_from_sdf


def list_from_file(file_path):
    """Converts [x, y, z] points from file into list"""
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
    """ Use clustering approach to find groups of points. Information supplements the automized labeling. """

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


def lidar_labeling(source_path, world_file, label_points=True, draw_bboxes=True):
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

        source_path --> path to the directory that contains the lidar and odometry data (described above)
        world_file --> file that contains all the information about the simulation the data is from (xml file)
    """

    CONE_RADIUS = 0.2  # in [m]

    # set/get koopacar pose and known cones positions
    if world_file is None:
        koopacar_start_yaw = 0  # euler angle in radians
        koopercar_start_pos = np.array([0, 0])  # [x, y] in [m]

        cone_world_positions = np.array([np.array(c) for c in
                                         list_from_file(os.path.join(source_path, "cone_pos.txt"))])

    else:
        world_xml = world_file.read()
        koopacar_pose = bot_pose_from_sdf(world_xml)

        koopacar_start_yaw = koopacar_pose[-1]  # euler angle in radians
        koopercar_start_pos = np.array([koopacar_pose[0], koopacar_pose[1]])  # [x, y] in [m]

        cone_world_positions = cone_position_from_sdf(world_xml)

    # known cone pos relative to koopercar pos
    cone_positions = rotation(translation(cone_world_positions[::, 0:2], -koopercar_start_pos), koopacar_start_yaw)

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
        relative_cones = rotation(translation(cone_positions, - delta_pos), - delta_orientation)

        # collect ranges from current scan and calculates points
        with open(os.path.join(source_path, "lidar_scan", scan_file), "r") as ranges_file:
            ranges = [float(data) for data in ranges_file]
        # points = np.array([np.array(c) for c in list_from_file(os.path.join(source_path, "lidar_scan", scan_file))])

        # draw bboxes
        if draw_bboxes:
            _draw_bboxes(source_path, scan_file, relative_cones, CONE_RADIUS)

        # label points
        if label_points:
            _label(source_path, scan_file, ranges, relative_cones, CONE_RADIUS)


def _label(source_path, scan_file, ranges, relative_cones, cone_radius):
    """
    Labels given points with DBSCAN and ground truth

    No cone label: 0
    Cone label: 1
    Outlier/inf ranges: 2

    source_path    --> root-folder (described in lidar_labeling())
    scan_file      --> name of currently used scan file
    ranges         --> ranges od lidar scan
    relative_cones --> ground truth cone coordinates in [x, y, z]
    cone_radius    --> radius of a cone
    """
    # define labels
    NO_CONE_LABEL = 0
    CONE_LABEL = 1
    OUTLIER_LABEL = 2

    # find clusters
    EPSILON = 0.15
    MIN_SAMPLES = 5

    # calculate points from ranges
    points = lidar_data_to_point(ranges)

    cluster_labels = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit_predict(points)

    # create/open label file
    filename_label = os.path.join(source_path, "label", scan_file.replace(".txt", "") + ".label")
    with open(filename_label, "w") as label_file:
        # loop over all points from scan
        for p_ind, point in enumerate(points):
            # return if points is at [0, 0] -> outlier/inf
            if point[0] == 0 and point[1] == 0:
                label_file.write(f"{OUTLIER_LABEL}\n")
                continue

            # loop over all known cones
            cone_hit = False
            for cone in relative_cones:

                # check if points are inside cone base area: (x - center_x)² + (y - center_y)² < radius²
                left = pow((point[0] - cone[0]), 2) + pow((point[1] - cone[1]), 2)
                right = pow(cone_radius, 2)
                if left < right and cluster_labels[p_ind] != -1:
                    cone_hit = True
                    break

            # write label to file
            if cone_hit:
                label_file.write(f"{CONE_LABEL}\n")
            else:
                label_file.write(f"{NO_CONE_LABEL}\n")


def _draw_bboxes(source_path, scan_file, relative_points, cone_radius):
    """
    Draws bounding boxes with ground truth

    source_path    --> root-folder (described in lidar_labeling())
    scan_file      --> name of currently used scan file
    relative_cones --> ground truth cone coordinates in [x, y, z]
    cone_radius    --> radius of a cone
    """
    # create/open label file
    filename_bbox = os.path.join(source_path, "bboxes", scan_file.replace(".txt", "") + ".bbox")
    with open(filename_bbox, "w") as bbox_file:
        # loop over all cones
        for cone in relative_points:
            bbox_file.write(f"1 {cone[0]: .3f} {cone[1]: .3f} {cone_radius:.3f} {cone_radius:.3f}\n")


def main(args=None):
    PATH_TO_SOURCE = "../../data/lidar_perception/new_data_set"

    try:
        world_file_path = "/home/ubuntu/koopacar-simulation-assets/src/koopacar_simulation/koopacar_simulation/" \
                          "worlds/cone_cluster.world"
        world_file = open(world_file_path, 'r')

        lidar_labeling(PATH_TO_SOURCE, world_file)

        world_file.close()

    except OSError:
        print("World file was not found. Information about the simulation will not be used in the calculations. \n"
              "The cone positions will be taken from the txt file in the directory and the bot pose will be zero.")

        lidar_labeling(PATH_TO_SOURCE, None)


if __name__ == '__main__':
    main()
