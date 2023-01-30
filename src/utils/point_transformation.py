from numpy import inf
import numpy as np


def lidar_data_to_point(ranges):
    """
    Converts ranges into coordinates.

    Ranges are indexed by angle, and describe the distance until the lidar hit an object.
    Points are returned in array as coordinates in format [x, y]. (Note: the coordinates refer to
    cartesian coordinates.)
    """
    points_x = np.array(ranges) * np.sin(np.flip(np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)))
    points_y = np.array(ranges) * np.cos(np.flip(np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)))
    points = np.array([[x, y] for x, y in zip(points_x, points_y)])

    return points


def remove_inf_point(points):
    """ Remove points in infinity from list. Returns numpy array of points without inf-points. """

    inf_index = []

    for i in range(0, points.shape[0]):
        if points[i, 0] == inf or points[i, 0] == -inf:
            inf_index.append(i)

    return np.delete(points, inf_index, axis=0)


