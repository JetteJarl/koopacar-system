from sklearn.cluster import DBSCAN
import numpy as np


CONE_LABEL = 1
CONE_RADIUS = 0.2


def get_cone_centroids(labels, points, ranges):
    cone_points = points[labels == CONE_LABEL]
    cone_ranges = ranges.reshape(-1, )[labels == CONE_LABEL]
    cluster_labels_all = DBSCAN(eps=0.1, min_samples=3).fit_predict(cone_points)

    cones = []

    for index, label in enumerate(np.unique(cluster_labels_all)):
        if label == -1:
            continue

        cluster_points = cone_points[cluster_labels_all == label]
        cluster_ranges = cone_ranges[cluster_labels_all == label]

        closest = cluster_points[np.where(cluster_ranges == np.min(cluster_ranges))][0]

        scalar = CONE_RADIUS / (np.min(cluster_ranges) + 1e-9)
        center = closest + scalar * closest

        cones.append([center, cluster_points])

    return cones
