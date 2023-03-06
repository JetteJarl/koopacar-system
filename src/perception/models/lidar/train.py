import os
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras

from src.perception.models.lidar.lidar_cnn import create_model

data_dir = "/home/ubuntu/koopacar-system/data/lidar_perception/training_data/lidar_03"
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

model = create_model()

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.Recall(),
             tf.keras.metrics.Precision()]
)

history = model.fit(x_train, y_train, batch_size=16, epochs=64, validation_data=(x_test, y_test))

model.save_weights("./weights/")
