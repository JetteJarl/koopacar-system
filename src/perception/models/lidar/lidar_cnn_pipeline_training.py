import math
import os
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import mlflow
import argparse

from src.perception.models.lidar.lidar_cnn import *
from src.perception.models import pipeline_utils

WEIGHTS_PATH = "./weights"
MODEL_PATH = "./model"

LOSS = 0


def use_end_loss(y_true, y_pred):
    loss = LOSS**2

    return loss


class LidarCNN:
    def __init__(self, batch_size, data_path):
        # self.model = create_model(loss_function=use_end_loss)
        self.model = create_model()

        self.x_in_batches, self.y_in_batches = self._load_data_as_batches(batch_size, data_path)

    def _load_data_as_batches(self, bs, data_path):
        # TODO: Change to relative + input
        data_dir = '/home/ubuntu/koopacar-system/data/perception/training_data/complete_04'
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

        number_batches = math.floor(X.shape[0] / bs)

        x = np.array_split(X[:(number_batches*bs)], number_batches)
        y = np.array_split(Y[:(number_batches*bs)], number_batches)

        if X.shape[0] % bs != 0:
            x.append(X[(number_batches*bs):])
            y.append(X[(number_batches*bs):])

        return x, y

    def forward(self, batch, loss_function=pipeline_utils.cross_entropy):
        y_pred = self.model.predict(self.x_in_batches[batch])

        loss = loss_function(self.y_in_batches[batch], y_pred)

        return y_pred, loss

    def backward(self, batch, loss):
        LOSS = loss

        self.model.fit(self.x_in_batches[batch], self.y_in_batches[batch],
                       epochs=1, batch_size=self.x_in_batches[batch].shape[0])

    def train(self):
        pass

    def save_model(self):
        self.model.save(MODEL_PATH)

    def save_weight(self):
        self.model.save_weights(WEIGHTS_PATH)
