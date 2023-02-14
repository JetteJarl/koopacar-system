import numpy as np


def lidar_matrix(points, resolution=(256, 256), lidar_range=3.5):
    """
    Creates image/matrix matching the top-down view of point cloud described by points.

    points --> points gathered from Lidar scan, representation of point cloud in FLU
    optional:
    resolution --> resolution of the target matrix/image
    lidar_range --> max range of the lidar being used
    """
    # TODO: Find more efficient implementation
    points = np.array(points)
    point_matrix = np.zeros(resolution)

    step_size_x = lidar_range * 2 / resolution[0]
    step_size_y = lidar_range * 2 / resolution[1]

    y_upper = lidar_range
    y_lower = y_upper - step_size_y

    for row_ind in range(0, resolution[0]):
        x_upper = lidar_range
        x_lower = x_upper - step_size_x

        for column_ind in range(0, resolution[1]):
            is_in_x_range = np.logical_and(points[:, 0] < x_upper, points[:, 0] >= x_lower)
            is_in_y_range = np.logical_and(points[:, 1] < y_upper, points[:, 1] >= y_lower)

            for is_x, is_y in zip(is_in_x_range, is_in_y_range):
                if is_x and is_y:
                    point_matrix[row_ind, column_ind] = 1

            x_upper -= step_size_x
            x_lower -= step_size_x

        y_upper -= step_size_y
        y_lower -= step_size_y

    return point_matrix


class LidarCNN:
    def __init__(self):
        pass

    def gather_input(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass
