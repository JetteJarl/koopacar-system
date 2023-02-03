import matplotlib.pyplot as plt
import numpy as np


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
    try:
        data_input = open("../../data/new_data_set/" + args, "r")
        data = []
        for line in data_input:
            if line == "\n":
                continue

            point = list(map(float, line[1:len(line) - 2].split(",")))
            data.append(point)

        plot_raw_points_3d(data)
    except OSError:
        print("Could not open/read file: " + args)


if __name__ == '__main__':
    main("lidar_scan20230130-112604.txt")
