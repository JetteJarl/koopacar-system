import matplotlib.pyplot as plt
import numpy as np

from src.utils.point_transformation import convert_FLU_to_ENU
from src.utils.point_transformation import convert_ENU_to_FLU
from src.utils.point_transformation import translation


def plot_raw_points_3d(data):
    data = np.array(data)

    plt.scatter(data[::, 0], data[::, 1], s=0.5)

    plt.show()


def plot_labled_data_3d(data, labels, cone_label=1, title='', xlim=(-4, 4), ylim=(-4, 4)):
    data = np.array(data)

    color = ['red' if label == cone_label else 'grey' for label in labels]

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=0.5)

    plt.title(title)
    plt.show()


def main(args=None):
    # lidar scan
    scan_input = open("../../data/lidar_perception/new_data_set/lidar_scan/" + args + ".txt", "r")
    scan = []
    scan_and_cones = []

    for line in scan_input:
        if line == "\n":
            continue

        point = list(map(float, line[1:len(line) - 2].split(",")))
        scan.append(point)
        scan_and_cones.append(point)

    # corresponding labels
    label_input = open("../../data/lidar_perception/new_data_set/label/" + args + ".label", "r")
    labels = []
    for line in label_input:
        if line == "\n":
            continue

        label = int(line[0])
        labels.append(label)

    # cones
    cones_input = open("../../data/lidar_perception/new_data_set/cone_pos.txt", "r")
    cones = []
    for line in cones_input:
        if line == "\n":
            continue

        temp1 = line[1:len(line) - 2].split(",")
        temp = map(float, line[1:len(line) - 2].split(","))
        cone = list(map(float, line[1:len(line) - 2].split(",")))
        cones.append(cone)

    # cones = convert_FLU_to_ENU(cones)
    for cone in cones:
        scan_and_cones.append(cone)

    # TODO: cleanup
    # plot_raw_points_3d(scan_and_cones)
    plot_labled_data_3d(scan, labels)
    if True:
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        scan = np.array(scan)
        cones = np.array(cones)
        # cones = translation(cones, convert_ENU_to_FLU([[-0, -0, 0]]))
        plt.scatter(scan[:, 0], scan[:, 1], s=0.5)
        plt.scatter(cones[:, 0], cones[:, 1], color="red", s=1)

        plt.title("")
        plt.show()


if __name__ == '__main__':
    main("lidar_scan_20230206-132801")
