import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.utils.plot_data import plot_labled_data_3d


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


def lidar_labeling_bbox():
    pass


def main(args=None):
    PATH_TO_DATA = "../../data/lidar_perception/new_lidar_set"
    PATH_TO_LABEL = "../../data/label"

    for file in sorted(os.listdir(PATH_TO_DATA)):
        try:
            data_input = open(os.path.join(PATH_TO_DATA, file), "r")
            data = []
            for line in data_input:
                if line == "\n":
                    continue

                point = list(map(float, line[1:len(line) - 2].split(",")))
                data.append(point)

            labels = lidar_labeling_dbscan(np.array(data))

            label_file = open(os.path.join(PATH_TO_LABEL, os.path.splitext(file)[0] + ".label"), 'w')
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
