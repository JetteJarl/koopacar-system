import os
from sklearn.cluster import DBSCAN
import numpy as np
import re

from src.utils.ros2_message_parser import string2odom
from src.utils.point_transformation import *
from src.utils.parse_from_sdf import *
from src.utils.file_operations import *


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
    """ Uses the know position of cones to create bboxes/label the points.

        Warning: If there are old lidar scans that are in the directory that don't match the current setup
        (meaning bot pose) make sure to save them in a different directory.
        Any scan in the directory will be labeled using the current setup. The correctness of the positions needs to
        be verified by the user.
        The labeling only works correctly if the bot is not moved during the scans. After every movement the bot pose
        needs to be updated manually and the script run again on the data collected after the movement.

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
        | -- bot_pose.txt

        source_path --> path to the directory that contains the lidar and odometry data (described above)
        world_file --> file that contains all the information about the simulation the data is from (xml file)
    """

    CONE_RADIUS = 0.25  # in [m]

    # set/get koopacar pose and known cones positions
    koopacar_pose = np.fromstring(open(os.path.join(source_path, "bot_pose.txt")).read(), dtype=float, sep=' ')

    koopacar_start_yaw = koopacar_pose[-1]  # euler angle in radians
    koopercar_start_pos = np.array([koopacar_pose[0], koopacar_pose[1]])  # [x, y] in [m]

    world_xml = world_file.read()
    cone_world_positions = cone_position_from_sdf(world_xml)

    # known cone pos relative to koopercar pos
    cones = rotation(translation(cone_world_positions[::, 0:2], -koopercar_start_pos), -koopacar_start_yaw)

    # load lidar scan data
    path_to_scan = os.path.join(source_path, "lidar_points")
    all_scan_files = sorted(os.listdir(path_to_scan))

    # loop over all lidar scan files, to label
    for index, scan_file in enumerate(all_scan_files):
        points = np.array([np.array(c) for c in list_from_file(os.path.join(source_path, "lidar_points", scan_file))])

        # draw bboxes
        if draw_bboxes:
            _draw_bboxes(source_path, scan_file, cones, CONE_RADIUS)

        # label points
        if label_points:
            _label(source_path, scan_file, points, cones, CONE_RADIUS)


def _label(source_path, scan_file, points, relative_cones, cone_radius):
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
    # points = lidar_data_to_point(points)

    cluster_labels = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit_predict(points)

    # create/open label file
    filename_label = os.path.join(source_path, "label", scan_file.replace(".bin", "") + ".label")
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
    filename_bbox = os.path.join(source_path, "bboxes", scan_file.replace(".bin", "") + ".bbox")
    with open(filename_bbox, "w") as bbox_file:
        # loop over all cones
        for cone in relative_points:
            bbox_file.write(f"1 {cone[0]: .3f} {cone[1]: .3f} {cone_radius:.3f} {cone_radius:.3f}\n")


def main(args=None):
    PATH_TO_SOURCE = "../../data/lidar_perception/new_data_set"

    world_file_path = "/home/ubuntu/koopacar-simulation-assets/src/koopacar_simulation/koopacar_simulation/" \
                          "worlds/cone_cluster.world"
    world_file = open(world_file_path, 'r')

    lidar_labeling(PATH_TO_SOURCE, world_file)

    world_file.close()


if __name__ == '__main__':
    main()
