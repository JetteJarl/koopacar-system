import cv2
import numpy as np
import os
import math

import torch

from src.localization.fusion import detect_cones
from src.perception.models.lidar.cones_from_prediction import CONE_LABEL, CONE_RADIUS, get_cone_centroids
from src.perception.models.lidar.lidar_cnn import probability_to_labels
from src.utils.point_transformation import lidar_data_to_point


def cross_entropy(Y, Y_pred):
    losses = []
    for y, y_pred in zip(Y, Y_pred):
        N = y.shape[0]
        loss_in_scan = -np.sum(y*np.log(y_pred + 1e-9))/N
        losses.append(loss_in_scan)

    return np.array(losses).mean()


def mse(Y, Y_pred):
    losses = []
    for y, y_pred in zip(Y, Y_pred):
        N = y.shape[0]
        loss_in_scan = np.sum(np.square((y - y_pred))) / N
        losses.append(loss_in_scan)

    return np.array(losses).mean()


def intersect(A, B):
    d = np.linalg.norm(A - B, 2)
    circle_area = math.pi * CONE_RADIUS ** 2

    if d == 0:
        return circle_area

    if d > CONE_RADIUS:
        return 0

    angle = d / (2 * CONE_RADIUS**2)
    theta = math.acos(angle) * 2
    area_ = (0.5 * theta * CONE_RADIUS**2) - (0.5 * CONE_RADIUS**2 * math.sin(theta))

    return 2*area_


def calc_pos_loss(A, B):
    area = math.pi * CONE_RADIUS ** 2
    intersect_area = intersect(A, B)

    union = 2*area - intersect_area

    return 1 - intersect_area / union




def calc_cls_loss(A, B):
    pass


def load_bounding_boxes(data_path):
    """
        Load bounding boxes / yolov5 labels / expected output.

        returns the bounding boxes divided in batches
    """
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

    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes


def parse_pipeline_ground_truth(data_path, bounding_boxes):
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
        cones_in_sim, used_bboxes = detect_cones(bboxes, cones)
        ground_truth.append((cones_in_sim, used_bboxes))

    return ground_truth


def compute_cluster_loss(predicted_clusters, lidar_x_ranges, lidar_y):
    """
        Computes the loss produced by the lidar-cnn using only the end prediction.

        To this end, it uses the points that are related to a bounding box and cone centroid to calculate
        how these points diverge from the ground truth.
    """
    lidar_x_points = lidar_data_to_point(lidar_x_ranges)
    compare_index = np.where(np.in1d(lidar_x_points, predicted_clusters))[0]

    true_end_lidar = lidar_y[compare_index]

    probability_vector_cone = np.zeros((1, 3))
    probability_vector_cone[CONE_LABEL] = 1

    pred_end_lidar = np.full_like(true_end_lidar, probability_vector_cone)

    return mse(true_end_lidar, pred_end_lidar)


def training_fusion(lidar_pre, lidar_y, lidar_x, yolo_path, img_list, data_path, ground_truth):

    yolov5_model = torch.hub.load('../../../../../yolov5', 'custom', yolo_path, source='local')

    lidar_points = np.array([lidar_data_to_point(x.reshape(x.shape[0],)) for x in lidar_x])
    lidar_labels = probability_to_labels(lidar_pre)

    end_pos_loss = 0
    end_cls_loss = 0

    for l_points, l_ranges, l_pre, l_y, img, g in zip(lidar_points, lidar_x, lidar_labels, lidar_y, img_list, ground_truth):
        # get yolov5 prediction
        img = cv2.imread(os.path.join(data_path, 'img/images', img))[:, :, ::-1]
        bboxes = yolov5_model(img).xyxy[0]
        centroids = get_cone_centroids(l_pre, l_points, l_ranges)

        # fusion
        if len(centroids) == 0:
            cones, used_bboxes = np.array([]), np.array([])
        else:
            cones, used_bboxes = detect_cones(bboxes.cpu().detach().numpy(), centroids[:][0])

        # calculate loss
        if len(cones) == 0:     # in the case no prediciton was made
            end_pos_loss += len(g)
            end_cls_loss += len(g)  # TODO: What should be here actually

        if len(g) == 0:     # in the case there was no prediction to be made but a prediciton was made
            end_pos_loss += len(cones)
            end_cls_loss += len(cones)  # TODO: What should be here actually

        for c, u_bb in zip(cones, used_bboxes):

            # find closest cone
            current_dist = math.inf
            closest = None
            for true_c in g:
                dist = np.linalg.norm(true_c - c, 2)
                if dist <= current_dist:
                    closest = true_c
                    current_dist = dist

            end_pos_loss += calc_pos_loss(closest, c)
            end_cls_loss += calc_cls_loss(closest, c)

    return end_pos_loss / lidar_points.shape[0], end_cls_loss / lidar_points.shape[0]


