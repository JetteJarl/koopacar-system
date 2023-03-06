import os
import numpy as np

from src.perception.models.lidar.lidar_cnn import create_model
from src.utils.plot_data import plot_lidar_cnn_results

model = create_model()
model.load_weights("./weights/")


data_dir = "/home/ubuntu/koopacar-system/data/lidar_perception/test_data/lidar_01"
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


y_prediction = model.predict(X)

for i in range(0, len(y_prediction)):
    plot_lidar_cnn_results(X[i], y_prediction[i], Y[i])

