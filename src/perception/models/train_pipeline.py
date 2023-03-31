import math
import os
import numpy as np
import tensorflow as tf
import torch
import cv2
import mlflow

from src.perception.models.lidar import lidar_cnn_pipeline_training, lidar_cnn
from src.perception.models.camera import yolo_pipeline_training
from src.localization.localization.sensor_fusion_node import _detect_cones
from src.perception.models.lidar.lidar_cnn import probability_to_labels
from src.perception.perception.lidar_obj_detection_node import get_cone_centroids, CONE_RADIUS, CONE_LABEL

from src.perception.models.camera import train_backward
from src.perception.models.camera import train_forward
from src.utils.point_transformation import inf_ranges_to_zero, lidar_data_to_point


PATH_TO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..')
DATA_PATH = os.path.join(PATH_TO_ROOT, 'data/perception/training_data/complete_04')


def _load_lidar_data(data_path):
    # lidar data and labels
    lidar_scan_path = os.path.join(data_path, 'ranges')
    lidar_points_path = os.path.join(data_path, 'points')
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

    lidar_x_ranges = np.expand_dims(np.array(scans), axis=2)
    lidar_y = lidar_cnn.labels_to_probability(np.array(scan_labels))

    return lidar_x_ranges, lidar_y


def _parse_pipeline_ground_truth(data_path, bounding_boxes):
    cones_path = os.path.join(data_path, "cones")

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

    return all_cones, ground_truth


def _load_bounding_boxes(data_path):
    bounding_box_path = os.path.join(data_path, "img/labels")

    names = ['blue', 'orange', 'yellow']

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

    return np.array(bounding_boxes)


def _get_bounding_box_batch_tensor(bounding_box_batch_array):
    bounding_box_tensor = []

    for bbox_in_img in bounding_box_batch_array:
        for bbox in bbox_in_img:
            bounding_box_tensor.append(bbox)

    return torch.from_numpy(np.array(bounding_box_tensor).reshape(-1, 6))


def _load_images(data_path):
    image_path = os.path.join(data_path, "img/images")
    yolo_x_list = sorted(os.listdir(image_path))
    # shape = (len(yolo_x_list), )

    images = []

    for img_file in yolo_x_list:
        img = cv2.imread(os.path.join(image_path, img_file), flags=cv2.IMREAD_COLOR)
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        images.append(img)

    images_tensor = torch.from_numpy(np.array(images, dtype=float))

    return images_tensor


def _load_image_batch(data_path, img_list):
    image_path = os.path.join(data_path, "img/images")

    images = []

    for img_file in img_list:
        img = cv2.imread(os.path.join(image_path, img_file), flags=cv2.IMREAD_COLOR)
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        images.append(img)

    images_tensor = torch.from_numpy(np.array(images, dtype=float))

    return images_tensor


def _chunks_from_list(complete_list, batch_size):
    number_batches = math.ceil(len(complete_list) / batch_size)

    chunks = []
    for i in range(0, number_batches):
        start_index, end_index = i*batch_size, i*batch_size + batch_size
        chunks.append(complete_list[start_index:end_index])

    return chunks


def _cross_entropy(Y, Y_pred):
    losses = []
    for y, y_pred in zip(Y, Y_pred):
        N = y.shape[0]
        loss_in_scan = -np.sum(y*np.log(y_pred + 1e-9))/N
        losses.append(loss_in_scan)

    return np.array(losses).mean()


def _mse(Y, Y_pred):
    losses = []
    for y, y_pred in zip(Y, Y_pred):
        N = y.shape[0]
        loss_in_scan = np.sum(np.square((y - y_pred))) / N
        losses.append(loss_in_scan)

    return np.array(losses).mean()


def train_pipeline(data_path, save_path=os.path.join(PATH_TO_ROOT, 'models/yolov5/'), cfg=os.path.join(PATH_TO_ROOT, 'src/perception/models/camera/models/yolov5n.yaml'), epochs=64, batch_size=16):
    # Get data
    lidar_x_ranges, lidar_y = _load_lidar_data(data_path)
    yolo_x = _load_images(data_path)
    yolo_y = _load_bounding_boxes(data_path)
    cones, pipeline_y = _parse_pipeline_ground_truth(data_path, yolo_y)

    number_batches = math.ceil(lidar_x_ranges.shape[0] / batch_size)

    yolo_x_list = sorted(os.listdir(os.path.join(data_path, "img/images")))
    yolo_batches_x_list = np.array_split(np.array(yolo_x_list), number_batches)
    yolo_batches_y = np.array_split(yolo_y, number_batches)

    lidar_batches_x_ranges = np.array_split(lidar_x_ranges, number_batches)
    lidar_batches_y = np.array_split(lidar_y, number_batches)

    cones_batches = _chunks_from_list(cones, batch_size)
    pipeline_batches_y = np.array_split(pipeline_y, number_batches)

    # Create model
    lidar_cnn = lidar_cnn_pipeline_training.LidarCNN()
    yolo_pipeline_model = yolo_pipeline_training.YoloPipeline(cfg, save_path)

    # Log params
    mlflow.log_param('epochs', epochs)
    mlflow.log_param('batch_size', batch_size)

    # Start training
    for epoch in range(0, epochs):
        print(f"Epoch {epoch}:")

        lidar_total_loss = 0
        yolo_total_loss = 0
        yolo_total_box_loss = 0
        yolo_total_obj_loss = 0
        yolo_total_cls_loss = 0

        for batch_iteration in range(0, number_batches):
            print(f"Batch {batch_iteration} of {number_batches}: ")

            # lidar forward
            lidar_y_prediction, lidar_loss = lidar_cnn.forward(lidar_batches_x_ranges[batch_iteration], lidar_batches_y[batch_iteration], _cross_entropy)
            lidar_y_pred_label = probability_to_labels(lidar_y_prediction)

            lidar_total_loss += lidar_loss
            print(f"Lidar Loss -- {lidar_loss}")

            # yolo forward
            yolo_batch_x = _load_image_batch(data_path, yolo_batches_x_list[batch_iteration])
            yolo_batch_y_tensor = _get_bounding_box_batch_tensor(yolo_batches_y[batch_iteration])
            yolo_loss, yolo_loss_items = yolo_pipeline_model.forward(yolo_batch_x, yolo_batch_y_tensor)

            print(f"Yolov5 Loss -- total: {yolo_loss} (bbox_loss: {list(yolo_loss_items)[0]}, obj_loss: {list(yolo_loss_items)[1]}, cls_loss: {list(yolo_loss_items)[2]}")

            yolo_prediction = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(save_path, 'weights/last.pt'))(yolo_batch_x)
            bounding_boxes_prediction = yolo_prediction.cpu().detach().numpy()

            yolo_total_loss += list(yolo_loss)[0]
            yolo_total_box_loss += list(yolo_loss_items)[0]
            yolo_total_obj_loss += list(yolo_loss_items)[1]
            yolo_total_cls_loss += list(yolo_loss_items)[2]

            pipeline_y_pred = []

            # calc pipeline output
            for i in range(0, lidar_batches_x_ranges[batch_iteration].shape[0]):
                if np.any(lidar_y_pred_label[i] == CONE_LABEL):

                    predicted_labels_in_scan = lidar_y_pred_label[i]
                    ranges_in_scan = lidar_batches_x_ranges[batch_iteration][i]
                    points_in_scan = lidar_data_to_point(ranges_in_scan.reshape(-1,))

                    centroids_in_scan = get_cone_centroids(predicted_labels_in_scan, points_in_scan, ranges_in_scan)

                    result = _detect_cones(bounding_boxes_prediction[i], centroids_in_scan)

                    pipeline_y_pred.append(result)

            # TODO: Calc pipeline loss, does not have to be related to all centroids or all bounding boxes
            end_loss = None
            # mlflow.log_metric('end/loss', end_loss, epoch)

            # lidar backward
            lidar_cnn.backward(lidar_batches_x_ranges[batch_iteration], lidar_batches_y[batch_iteration], lidar_loss)

            # yolo backward
            yolo_pipeline_model.backward(yolo_loss, epoch, yolo_prediction)

        mlflow.log_metric('lidar/loss', lidar_total_loss, epoch)

        mlflow.log_metric('yolo/loss', yolo_total_loss, epoch)
        mlflow.log_metric('yolo/box_loss', yolo_total_box_loss, epoch)
        mlflow.log_metric('yolo/obj_loss', yolo_total_obj_loss, epoch)
        mlflow.log_metric('yolo/cls_loss', yolo_total_cls_loss, epoch)


def main(args=None):
    train_pipeline(DATA_PATH)


if __name__ == '__main__':
    main()
