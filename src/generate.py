import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import toml
import os
import uuid

import utils.load_data
import utils.preprocess_image
import utils.misc
import model

config = toml.load('./config/config.toml')
BATCH_SIZE = config['training']['batch_size']
IMAGE_DIMENSIONS = tuple(config['model']['image_dimensions'])

PATH_TO_PICTOGRAMS = config['paths']['pictograms']
pictograms_path = 'C:/Users/Frederik/Documents/Studienarbeit/data/Pictograms'
x_pictograms = tf.keras.utils.image_dataset_from_directory(pictograms_path, batch_size=BATCH_SIZE,
                                                           image_size=IMAGE_DIMENSIONS, labels=None)
x_pictograms_processed = utils.load_data.normalize_dataset(x_pictograms)

cycle_gan = model.CycleGan(image_size=IMAGE_DIMENSIONS)
cycle_gan.compile()
cycle_gan.restore_latest_checkpoint_if_exists()

# generate random images
for i in range(10):
    transformed_pictograms = x_pictograms_processed.map(
        lambda t: tf.py_function(utils.preprocess_image.randomly_transform_4d_tensor, inp=[t],
                                 Tout=tf.float32)
    )
    transformed_pictograms.shuffle(buffer_size=150, reshuffle_each_iteration=True)
    single_pictogram = transformed_pictograms.take(1).get_single_element()
    generator_result = cycle_gan.generator_g(single_pictogram)
    random_filename = str(uuid.uuid4())
    utils.misc.store_tensor_as_img(generator_result[0, :], random_filename, 'generator_test')
