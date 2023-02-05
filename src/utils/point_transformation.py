import copy
import math
from numpy import inf
import numpy as np


def translation(points, move_vector):
    """ Performs the translation indicated by the movement vector on the given set of points.

        points      --> arbitrary set of points
        move_vector --> vector indicating direction and length
    """
    np_points = np.array([np.array(p) for p in points])

    # fails if shapes dont match
    if np_points.shape[1] != move_vector.size:
        raise Exception("Shapes of move_vector and points are different")

    return np_points + move_vector


def rotation(points, rotation_angle, rotation_origin=(0, 0)):
    """ Rotates the given 2d points by degrees indicated by rotation_angle counterclockwise.

        points          --> arbitrary set of points
        rotation_angle  --> angle in radians
        rotation_origin --> single point [x, y]
    """
    np_points = np.array([np.array(p) for p in points])

    if points.shape[1] != 2:
        raise Exception("Points must be 2d: [x, y]")

    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle),  np.cos(rotation_angle)]])

    o = np.atleast_2d(rotation_origin)
    np_points = np.atleast_2d(np_points)

    return np.squeeze((R @ (np_points.T-o.T) + o.T).T)


def radians_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def lidar_data_to_point(ranges):
    """
    Converts ranges into coordinates.

    Ranges are indexed by angle, and describe the distance until the lidar hit an object.
    Points are returned in array as coordinates in format [x, y]. (Note: the coordinates refer to
    cartesian coordinates.)
    The value 'inf' in ranges can lead to unexpected behaviour of the function and might yield in nan
    """
    points_x = np.array(ranges) * np.sin(np.flip(np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)))
    points_y = np.array(ranges) * np.cos(np.flip(np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)))
    points = np.array([[x, y] for x, y in zip(points_x, points_y)])

    return points


def remove_inf_point(points):
    """ Remove points in infinity from list. Returns numpy array of points without inf-points. """

    inf_index = []

    for i in range(0, points.shape[0]):
        if points[i, 0] == inf or points[i, 0] == -inf \
                or points[i, 1] == inf or points[i, 1] == -inf:
            inf_index.append(i)

    return np.delete(points, inf_index, axis=0)


def remove_inf_ranges(ranges):
    """ Remove ranges that are inf. """
    ranges = copy.deepcopy(ranges)

    while inf in ranges:
        ranges.remove(inf)

    return ranges


def convert_FLU_to_ENU(coordinates):
    """Converts given set of coordinates from forward, left, up to east, north, up"""
    np_coordinates = np.array([np.array(c) for c in coordinates])

    enu_coordinates = np.array([np.array([-cord[1], cord[0], cord[2]]) for cord in np_coordinates])

    return enu_coordinates


def convert_ENU_to_FLU(coordinates):
    """Converts given set of coordinates from east, north, up to forward, left, up"""
    np_coordinates = np.array([np.array(c) for c in coordinates])

    flu_coordinates = np.array([np.array([cord[1], -cord[0], cord[2]]) for cord in np_coordinates])

    return flu_coordinates
