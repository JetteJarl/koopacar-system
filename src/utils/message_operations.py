import numpy as np


def header_to_float_stamp(header):
    """Converts the stamp of header to float."""
    return float(f"{header.stamp.sec}.{header.stamp.nanosec}")


def combine_secs_and_nsecs(secs, nsecs):
    return float(f"{int(secs)}.{int(nsecs)}")


def bbox_msg_to_values(bbox_msg):
    """
    Returns the data and time stamp of a bounding box message.

    Returns:
        bboxes: np.array of bounding boxes in format (top-left-x,
                top-left-y, bottom-right-x, bottom-right-y,
                confidence, class)
        bboxes_stamp: float of timestamp in format seconds.nanoseconds
    """
    # save data of message
    data = np.array(bbox_msg.data)
    data = data.reshape((-1, bbox_msg.layout.dim[1].size))

    # extract bounding boxes
    if data.size > 6:
        bboxes = data[1:]
    else:
        bboxes = np.array([])

    # extract time stamp of bounding boxes
    bboxes_stamp_sec = int(data[0, 0])
    bboxes_stamp_nano = int(data[0, 1])
    bboxes_stamp = float(f"{bboxes_stamp_sec}.{bboxes_stamp_nano}")

    return bboxes, bboxes_stamp


def centroid_msg_to_values(centroid_msg):
    """
    Returns the data and time stamp of a centroid message.

    Returns:
        centroids: np.array of centroids in format (x, y)
        centroids_stamp: float of timestamp in format seconds.nanoseconds
    """
    # TODO
    pass
