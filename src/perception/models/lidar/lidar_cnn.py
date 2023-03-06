import tensorflow as tf


def create_model():
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
    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu'))

    return model
