import os
from sklearn.cluster import DBSCAN
import argparse

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


def lidar_labeling(world_file, source_path, koopacar_pose, label_points=True, draw_bboxes=True):
    """ Uses the know position of cones to create bboxes/label the points.

        Warning: The labeling only works correctly if the bot is not moved during the scans. Make sure that the labeling
        is only run for a set of scans using the same position and that the koopacar_pose and world_file match the
        correct setup.

        Labels are saves in the following format:
        [label]
        Bounding boxes are saved in following format:
        [class] [center_x] [center_y] [height] [width]
        The directory source directory needs to have a directory called lidar_points containing the scans that need to
        be labeled. Other directory are created if they do not already exist.
        .
        | -- label
                | --
        | -- lidar_points
                | --
        | -- bboxes
                | --

        source_path     --> path to the directory that contains the lidar and odometry data (described above)
        world_file      --> file that contains all the information about the simulation the data is from (xml file) [sdf]
        koopacar_pose   --> pose of the bot recording the scans
    """
    CONE_RADIUS = 0.25  # in [m]

    # set/get koopacar pose and known cones positions
    koopacar_start_yaw = koopacar_pose[-1]  # euler angle in radians
    koopercar_start_pos = np.array([koopacar_pose[0], koopacar_pose[1]])  # [x, y] in [m]

    # TODO: Check usage if no cones in sim
    world_xml = world_file.read()
    cone_world_positions = cone_position_from_sdf(world_xml)

    # known cone pos relative to koopercar pos
    cones = rotation(translation(cone_world_positions[::, 0:2], -koopercar_start_pos), -koopacar_start_yaw)

    # load lidar scan data
    path_to_scan = os.path.join(source_path, "lidar_points")
    if not os.path.isdir(path_to_scan):
        print("lidar_points directory is missing. Ca not find scans.")
        return 2

    all_scan_files = sorted(os.listdir(path_to_scan))

    # loop over all lidar scan files, to label (only not labeled files)
    for index, scan_file in enumerate(all_scan_files):
        points = np.array([np.array(c) for c in list_from_file(os.path.join(source_path, "lidar_points", scan_file))])

        # draw bboxes
        if draw_bboxes:
            bbox_path = os.path.join(source_path, "bboxes")
            if not os.path.isdir(bbox_path):
                os.makedirs(bbox_path)

            bbox_file_path = os.path.join(bbox_path, scan_file.replace(".bin", ".bbox"))
            if not os.path.isfile(bbox_file_path):
                _draw_bboxes(bbox_file_path, cones, CONE_RADIUS)

        # label points
        if label_points:
            label_path = os.path.join(source_path, "label")
            if not os.path.isdir(label_path):
                os.makedirs(label_path)

            label_file_path = os.path.join(label_path, scan_file.replace(".bin", ".label"))
            if not os.path.isfile(label_file_path):
                _label(label_file_path, points, cones, CONE_RADIUS)


def _label(label_file_path, points, relative_cones, cone_radius):
    """
    Labels given points with DBSCAN and ground truth

    No cone label: 0
    Cone label: 1
    Outlier/inf ranges: 2

    label_path     --> file for labels
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

    # open label file
    with open(label_file_path, "w") as label_file:
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

    print(f"Created label for {label_file_path}")


def _draw_bboxes(bbox_file_path, relative_points, cone_radius):
    """
    Draws bounding boxes with ground truth

    bbox_path      --> destination for bboxes
    relative_cones --> ground truth cone coordinates in [x, y, z]
    cone_radius    --> radius of a cone
    """
    # open label file
    with open(bbox_file_path, "w") as bbox_file:
        # loop over all cones
        for cone in relative_points:
            bbox_file.write(f"1 {cone[0]: .3f} {cone[1]: .3f} {cone_radius:.3f} {cone_radius:.3f}\n")

    print(f"Created bounding_box for {bbox_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Script labeling a set lidar points from a koopacar-simulation.")
    parser.add_argument('-p', '-pose', nargs='+', type=float, help="Set of 6 numbers specifying a model pose in the "
                                                                   "gazebo simulation. Pose is specified as position "
                                                                   "[x, y, z] and orientation [x, y, z, w] (Quaternion)",
                        required=False)
    parser.add_argument('-w', '-world', type=str, help="File path specifying the location of the world file (sdf) "
                                                       "describing the simulation.", required=True)
    parser.add_argument('-d', '-dataset', type=str, help="Path to dataset.", required=True)
    parser.add_argument('-pf', '-posefile', type=str, help="File containing a set of possible bot poses. Pose is "
                                                           "specified as position [x, y, z] and orientation [x, y, z,"
                                                           " w] (Quaternion)",
                        required=False)
    parser.add_argument('-i', '-index', type=int, help="Index referencing a pose from the specified file.",
                        required=False)

    args = parser.parse_args()

    world_file = open(args.w, 'r')

    if args.p and args.pf and args.i:
        print("Please use only one of the options. Specify a pose (-p) OR a file and index (-pf, -i).")
        return 2

    if args.pf is not None and args.i is not None:
        pose_file = open(args.pf)
        all_poses = pose_file.readlines()

        pose = np.fromstring(all_poses[args.i].replace('\n', ''), dtype=float, sep=' ')

        assert len(pose) == 7
        roll_x, pitch_y, yaw_z = radians_from_quaternion(pose[3], pose[4], pose[5], pose[6])
        pose = np.array([pose[0], pose[1], pose[2], roll_x, pitch_y, yaw_z])

    elif args.p is not None:
        pose = np.array(args.p)

        assert len(pose) == 7
        roll_x, pitch_y, yaw_z = radians_from_quaternion(pose[3], pose[4], pose[5], pose[6])
        pose = np.array([pose[0], pose[1], pose[2], roll_x, pitch_y, yaw_z])

    else:
        print("Please specify a pose either directly or via a file. Use -h for more information.")
        return 2

    print(f"Using {pose} for labeling.")

    lidar_labeling(world_file, args.d, pose)

    world_file.close()


if __name__ == '__main__':
    main()
