import math
import numpy as np

# consts
CAMERA_FOV = 62
IMG_WIDTH = 640
IMG_HEIGHT = 480


def detect_cones(bboxes, centroids):
    """
    Returns detected cones from the bounding box and centroids data

    This method is what performs the sensor fusion.

    The returned cones is in format [x, y, label] and the used bounidng boxes
    """
    if len(centroids) == 0 or len(bboxes) == 0:
        return np.array([]), np.array([])

    # calculate angle of each centroid
    centroid_angles = [math.atan2(-centroid[1], centroid[0]) for centroid in centroids]
    centroid_angles = [math.degrees(centroid_angle) + CAMERA_FOV/2 for centroid_angle in centroid_angles]

    # sort centroids by distance to the bot/[0, 0]
    sorted_centroid_indices = np.argsort(np.linalg.norm(centroids, ord=2, axis=1))

    # bool map whether a centroid is assigned to a bounding box
    used_centroids = np.zeros((len(sorted_centroid_indices)))

    # fov to pixels ratio (width)
    fov_px_ratio = CAMERA_FOV / IMG_WIDTH

    # save labeled centroids
    labeled_centroids = []
    used_bboxes = []

    # iterate over bounding boxes, starting with the biggest/highest
    for bb in sorted(bboxes, key=lambda bb: bb[3] - bb[1], reverse=True):
        # approx angles of bounding box start/end
        start_angle = bb[0] * fov_px_ratio
        end_angle = bb[2] * fov_px_ratio

        # check for matching, not used centroids
        for idx in sorted_centroid_indices:
            if start_angle - 1 <= centroid_angles[idx] <= end_angle + 1 and not used_centroids[idx]:
                labeled_centroids.append((bb[5], centroids[idx]))
                used_bboxes.append(bb)
                used_centroids[idx] = 1
                break

    detected_cones = np.empty((len(labeled_centroids), 3))
    for i, (label, centroid) in enumerate(labeled_centroids):
        detected_cones[i] = centroid[0], centroid[1], label

    return detected_cones, used_bboxes
