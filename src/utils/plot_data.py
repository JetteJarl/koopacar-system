import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils.point_transformation import *
from src.utils.file_operations import *


def plot_raw_points_3d(data):
    data = np.array(data)

    plt.scatter(data[::, 0], data[::, 1], s=0.5)

    plt.show()


def plot_labled_data_3d(data, labels, cone_label=1, title='', xlim=(-4, 4), ylim=(-4, 4)):
    data = np.array(data)

    colors = ['black', 'red', 'lightgrey']
    color = [colors[round(label)] for label in labels]

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=0.5)

    plt.title(title)
    plt.show()


def subplots_labled_data_3d(data, labels, ax, cone_label=1):
    data = np.array(data)

    colors = ['black', 'red', 'lightgrey']
    color = [colors[round(label)] for label in labels]
    ax.scatter(data[:, 1], data[:, 0], c=color, s=0.5)


def plot_lidar_cnn_results(ranges, pred_labels, true_labels, xlim=(-4, 4), ylim=(-4, 4)):
    """
    Plots results from lidar_cnn prediction.

    ranges --> model input
    pred_labels --> labels returned from the prediction+
    true_labels --> actual labels
    """
    colors = ['black', 'red', 'lightgrey']

    color_pred = [colors[round(label[0])] for label in pred_labels]
    color_true = [colors[label] for label in true_labels]

    fig, axis = plt.subplots(1, 2)

    data = lidar_data_to_point(np.reshape(ranges, (360, )))

    axis[0].axis(xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])
    axis[0].scatter(data[:, 1], data[:, 0], c=color_true, s=0.5)

    axis[1].axis(xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])
    axis[1].scatter(data[:, 1], data[:, 0], c=color_pred, s=0.5)

    plt.title("Prediction (left: true, right: inferred)")
    plt.show()


def plot_label_verification(scan_path, label_path, cone_path):
    # lidar scan
    scan = list_from_file(scan_path)

    # corresponding labels
    label_input = open(os.path.join(label_path),"r")
    labels = np.fromstring(label_input.read().replace('\n', ' '), dtype=int, sep=' ')

    # cones
    cones = list_from_file(cone_path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    subplots_labled_data_3d(scan, labels, ax[0])

    ax[1].scatter(scan[:, 1], scan[:, 0], c='black', s=0.5)
    ax[1].scatter(cones[:, 1], cones[:, 0], c='red')

    fig.suptitle(f"Data and Labeling ({os.path.basename(scan_path)})")

    plt.show()


def main(args=None):
    data_dir = "../../data/lidar_perception/new_data_set/"

    all_scans = sorted(os.listdir(os.path.join(data_dir, "lidar_points")))
    all_labels = sorted(os.listdir(os.path.join(data_dir, "label")))

    for i in range(85, 150):
        scan_file = all_scans[i]
        label_file = all_labels[i]

        plot_label_verification(
            scan_path=os.path.join(data_dir, "lidar_points", scan_file),
            label_path=os.path.join(data_dir, "label", label_file),
            cone_path="../../data/lidar_perception/new_data_set/cone_pos.txt")


if __name__ == '__main__':
    main()
