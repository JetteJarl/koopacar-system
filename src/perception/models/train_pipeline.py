import os
import numpy as np

from src.perception.models.lidar import lidar_cnn_pipeline_training, lidar_cnn
from src.perception.models.camera import yolo_pipeline_training
from src.localization.localization.sensor_fusion_node import _detect_cones

DATA_PATH = '/home/ubuntu/koopacar-system/data/perception/training_data/complete_04'


def _parse_lidar_data(data_path):
    # lidar data and labels
    lidar_scan_path = os.path.join(data_path, 'ranges')
    lidar_labels_path = os.path.join(data_path, 'label')

    all_scans = sorted(os.listdir(lidar_scan_path))
    scans = []
    for scan_file in all_scans:
        if scan_file == ".gitkeep":
            continue

        with open(os.path.join(lidar_scan_path, scan_file)) as file:
            range_string = file.read().replace('\n', ' ')
            beam_range = np.fromstring(range_string, dtype=float, sep=' ')
            scans.append(beam_range)

    all_scan_labels = sorted(os.listdir(lidar_labels_path))
    scan_labels = []

    for label_file in all_scan_labels:
        if label_file == ".gitkeep":
            continue

        with open(os.path.join(lidar_labels_path, label_file)) as file:
            label_string = file.read().replace('\n', ' ')
            label = np.fromstring(label_string, dtype=int, sep=' ')
            scan_labels.append(np.array(label))

    lidar_x = np.expand_dims(np.array(scans), axis=2)
    lidar_y = lidar_cnn.labels_to_probability(np.array(scan_labels))

    return lidar_x, lidar_y


def _parse_pipeline_ground_truth(data_path):
    bounding_box_path = os.path.join(data_path, "img/labels")
    cones_path = os.path.join(data_path, "cones")

    # parse bounding boxes
    all_bbox_files = sorted(os.listdir(bounding_box_path))
    bounding_boxes = []

    for bbox_file in all_bbox_files:
        if bbox_file == ".gitkeep":
            continue

        with open(os.path.join(bounding_box_path, bbox_file)) as file:
            bboxes_in_img = []

            for line in file:
                bbox_string = line.replace('\n', ' ')
                bbox = np.fromstring(bbox_string, dtype=float, sep=' ')

                bbox = np.r_[bbox, 1]
                permutation = [1, 2, 3, 4, 5, 0]
                bbox = bbox[permutation]

                bboxes_in_img.append(bbox)

            bounding_boxes.append(np.array(bboxes_in_img))


    # parse cones
    all_cone_files = sorted(os.listdir(cones_path))
    all_cones = []

    for cone_file in all_cone_files:
        if cone_file == ".gitkeep":
            continue

        with open(os.path.join(cones_path, cone_file)) as file:
            cones_in_iteration = []
            for line in file:
                cone_string = line.replace('\n', ' ')
                cone = np.fromstring(cone_string, dtype=float, sep=' ')
                cones_in_iteration.append(cone)

            all_cones.append(np.array(cones_in_iteration))

    # determine ground truth
    ground_truth = []

    for bboxes, cones in zip(bounding_boxes, all_cones):
        cones_in_sim = _detect_cones(bboxes, cones)
        ground_truth.append(_detect_cones(bboxes, cones))

    return bounding_boxes, all_cones, np.array(ground_truth)


def train_pipeline(data_path, img_data_yaml):
    epochs = 1

    lidar_x, lidar_y = _parse_lidar_data(data_path)
    yolo_y, cones, pipeline_y = _parse_pipeline_ground_truth(data_path)

    lidar_cnn = lidar_cnn_pipeline_training.LidarCNN()

    for epoch in range(0, epochs):
            lidar_loss = lidar_cnn.forward(lidar_x, lidar_y)
            yolo_loss = yolo_pipeline_training.forward(data_set)

            # log losses
            # cone centers

            # fusion
            # log losses

            # propagate back
            # lidar_cnn.backward(data_sample)


def main(args=None):
    train_pipeline(DATA_PATH, "complete_04.yaml")


if __name__ == '__main__':
    main()
