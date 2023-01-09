import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import toml
import os

import utils.load_data
import utils.preprocess_image
import utils.misc
import model

config = toml.load('./config/config.toml')

BATCH_SIZE = config['training']['batch_size']
BUFFER_SIZE = 1000
IMAGE_DIMENSIONS = tuple(config['model']['image_dimensions'])

PATH_TO_TRAIN_DIRECTORY = config['paths']['train_data']
training_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_TRAIN_DIRECTORY))

x_train = tf.keras.utils.image_dataset_from_directory(training_path, batch_size=BATCH_SIZE, image_size=IMAGE_DIMENSIONS,
                                                      labels=None)
x_train_processed = utils.load_data.normalize_dataset(x_train)

PATH_TO_PICTOGRAMS = config['paths']['pictograms']
pictograms_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_PICTOGRAMS))

x_pictograms = tf.keras.utils.image_dataset_from_directory(pictograms_path, batch_size=BATCH_SIZE,
                                                           image_size=IMAGE_DIMENSIONS, labels=None)
x_pictograms_processed = utils.load_data.normalize_dataset(x_pictograms)

# x_pictograms = x_pictograms.map(lambda x: tf.py_function(utils.preprocess_image.randomly_transform_image,
#                                                         inp=[x], Tout=tf.float32))

# x_pictogram = x_pictograms.get_single_element()
# print(tf.shape(x_pictogram))
# x_pictogram = utils.preprocess_image.randomly_transform_4d_tensor(x_pictogram, IMAGE_DIMENSIONS)
# plt.imshow(x_pictogram[0, :])
# plt.show()

cycle_gan = model.CycleGan(image_size=IMAGE_DIMENSIONS)
cycle_gan.compile()
cycle_gan.restore_latest_checkpoint_if_exists()

# generate random images
#for i in range(10):
#    x_pictograms_transformed = x_pictograms_processed.map(lambda t: tf.py_function(
#        utils.preprocess_image.randomly_transform_4d_tensor,
#        inp=[t],
#        Tout=tf.float32))
#    generator_result = cycle_gan.generator_g(x_pictograms_transformed.get_single_element())
#    random_filename = str(uuid.uuid4())
#    utils.misc.store_tensor_as_img(generator_result[0, :], random_filename, 'generator_test')

def run_training():
    cycle_gan.print_welcome()
    # generator_test_result = cycle_gan.generator_g(x_pictograms.get_single_element()) # USE TEST DATA NOT TRAIN DATA
    cycle_gan.fit(x_pictograms_processed, x_train_processed, epochs=config['training']['number_of_epochs'])
    return
