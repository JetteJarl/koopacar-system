import os
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import mlflow
import argparse

from src.perception.models.lidar.lidar_cnn import *

WEIGHTS_PATH = "./weights"
MODEL_PATH = "./model"


class LidarCNN:
    def __init__(self):
        self.current_end_loss = 1
        self.current_lidar_loss = 1

        self.model = create_model()

    def _use_end_loss(self, y_true, y_pred):
        return self.current_end_loss

    def forward(self, x, y, loss_function):
        y_pred = self.model.predict(x)

        loss = loss_function(y, y_pred)

        return y_pred, loss

    def backward(self, x, y, loss):
        self.current_end_loss = loss

        x_tensor = tf.convert_to_tensor(x)
        y_tensor = tf.convert_to_tensor(y)

        data = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

        self.model.fit(x, y, epochs=1, batch_size=x.shape[0])

    def save_model(self):
        self.model.save(MODEL_PATH)

    def save_weight(self):
        self.model.save_weights(WEIGHTS_PATH)
