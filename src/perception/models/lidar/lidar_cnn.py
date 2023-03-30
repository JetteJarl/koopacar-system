import tensorflow as tf
import numpy as np


def labels_to_probability(y, num_classes=3):
    """
    Takes a vector of labels and creates for every label a vector with probability for each label.

    y           --> labels; shape = (N, 360, ) or (N, 360, 1)
    num_classes --> number of classes that are possible
    returns: vector with probability for each label, the entry with index matching the label  should be 1 the rest 0.
             shape = (N, 360, num_classes)
    """
    Y = np.zeros((y.shape[0], y.shape[1], num_classes))

    for scan_index, scan_labels in enumerate(y):
        for label_index, label in enumerate(scan_labels):
            Y[scan_index, label_index, label] = 1

    return Y


def probability_to_labels(y):
    """
    Takes the output from the lidar_cnn which is a vector of probabilities the length of the number of possible classes.

    y --> vectors with probabilities; shape = (N, 360, num_classes)
    return: vectors with labels; shape = (N, 360)
    """

    Y = np.zeros((y.shape[0], y.shape[1]), dtype=int)

    for pred_index, prediction in enumerate(y):
        for vect_index, prob_vector in enumerate(prediction):
            label = np.where(prob_vector == np.max(prob_vector))[0][0]
            Y[pred_index, vect_index] = label

    return Y


def create_model(loss_function=tf.keras.losses.MeanSquaredError()):
    """ Creates a CNN with 1d convolutions that can be used to detect cones in lidar scans. """
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, input_shape=(360, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=3, kernel_size=3, padding='same', activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=[tf.keras.metrics.Recall(),
                 tf.keras.metrics.Precision()]
    )

    return model
