import os
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import mlflow
import argparse

from src.perception.models.lidar.lidar_cnn import *


def train(data_path, val_split=0.2, batch_size=32, epochs=64):
    mlflow.tensorflow.autolog()

    data_dir = "/data/perception/training_data/lidar_03"
    scans_dir = os.path.join(data_path, "ranges")
    label_dir = os.path.join(data_path, "label")

    all_scans = sorted(os.listdir(scans_dir))
    all_labels = sorted(os.listdir(label_dir))

    ranges = []  # shape --> N x 360
    for scan_file in all_scans:
        if scan_file == ".gitkeep":
            continue

        with open(os.path.join(scans_dir, scan_file)) as file:
            range_string = file.read().replace('\n', ' ')
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

    X = np.expand_dims(X, axis=2)  # Model Input --> shape = (N, 360, 1)
    if Y.shape != (X.shape[0], X.shape[1], 3):
        Y = labels_to_probability(Y)  # Model Output --> shape = (N, 360, 3)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=val_split, shuffle=False)

    model = create_model()

    model.summary()

    # TODO: Find correct setup for early stopping
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    # model.fit(x_train, y_train, batch_size=32, epochs=128, callbacks=callback, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    model.save("./src/perception/models/lidar/model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Reference a path to the data set as absolute path. -d / --data",
                        required=True)
    parser.add_argument('-V', '--validation_split', type=float, default=0.2, help="Specifies the share of data from "
                                                                                  "the training set used for "
                                                                                  "validation. -V / "
                                                                                  "--validation_split", required=False)
    parser.add_argument('-B', '--batch_size', type=int, default=32, help="Specify batch-size for training.",
                        required=False)
    parser.add_argument('-e', '--epochs', type=int, default=64, help="Specify number of epochs for training.",
                        required=False)

    args = parser.parse_args()

    mlflow.set_experiment("lidar-cnn")

    with mlflow.start_run():
        mlflow.log_param("val_split", args.validation_split)
        train(args.data, val_split=args.validation_split, batch_size=args.batch_size, epochs=args.epochs)


if __name__ == '__main__':
    main()
