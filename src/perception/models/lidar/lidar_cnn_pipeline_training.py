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

        self.model = create_model(loss_function=self._use_end_loss)

    def _use_end_loss(self, y_true, y_pred):
        return self.current_end_loss

    def forward(self, x, y, loss_function):
        y_pred = self.model.predict(x)

        self.current_lidar_loss = loss_function(y, y_pred)

        return y_pred, self.current_lidar_loss

    def backward(self, data, loss):
        self.model.train_step(data)

    def save_model(self):
        self.model.save(MODEL_PATH)

    def save_weight(self):
        self.model.save_weights(WEIGHTS_PATH)
