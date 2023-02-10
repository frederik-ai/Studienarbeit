from PIL import Image
import os
import tensorflow as tf


def load_test_data(path_to_test_directory, batch_size=32, image_dimensions=(128, 128)):
    test_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', path_to_test_directory))
    x_test = tf.keras.utils.image_dataset_from_directory(test_path, batch_size=batch_size, image_size=image_dimensions,
                                                         labels=None)
    return x_test


def normalize_dataset(dataset):
    return dataset.map(lambda tensor: (tensor / 127.5) - 1)