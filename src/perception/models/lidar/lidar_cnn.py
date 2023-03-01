import numpy as np
import tensorflow as tf
import sklearn.model_selection
import os

from src.utils.file_operations import *
from src.utils.plot_data import plot_lidar_cnn_results


class LidarCNN:
    def __init__(self, scan_length=360):
        pass

    def __gather_input(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


def main():
    data_dir = "/home/ubuntu/koopacar-system/data/lidar_perception/training_data/lidar_01"
    scans_dir = os.path.join(data_dir, "ranges")
    label_dir = os.path.join(data_dir, "label")

    all_scans = sorted(os.listdir(scans_dir))
    all_labels = sorted(os.listdir(label_dir))

    ranges = []  # shape --> N x 360
    for scan_file in all_scans:
        if scan_file == ".gitkeep":
            continue

        with open(os.path.join(scans_dir, scan_file)) as file:
            range_string = file.read().replace('\n', ' ')
            # beam_range = np.array([np.array(entry) for entry in np.fromstring(range_string, dtype=float, sep=' ')])
            beam_range = np.fromstring(range_string, dtype=float, sep=' ')
            ranges.append(beam_range)

    labels = []
    for label_file in all_labels:
        if label_file == ".gitkeep":
            continue

        with open(os.path.join(label_dir, label_file)) as file:
            label_string = file.read().replace('\n', ' ')
            label = np.fromstring(label_string, dtype=int, sep=' ')
            labels.append(np.array(label))

    X = np.array(ranges)
    Y = np.array(labels)
    X = np.expand_dims(X, axis=2)  # Model Input --> shape = (N, 360, 1) ???
    # Y = np.expand_dims(Y, axis=2)  # Model Output --> shape = (N, 360, 1) ???

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=False)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(360, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu'))
    # TODO: Add dense (?)

    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.Precision()]
    )

    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    y_prediction = model.predict(x_test)

    for i in range(0, len(x_test)):
        plot_lidar_cnn_results(x_test[i], y_prediction[i], y_test[i])


if __name__ == '__main__':
    main()
