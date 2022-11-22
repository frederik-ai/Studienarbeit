import tensorflow as tf
import os
import matplotlib.pyplot as plt
from generator_model import make_generator_model
from discriminator_model import make_discriminator_model
from utils.preprocess_image import normalize, normalize_tensor_of_images
from configparser import ConfigParser
import model

# config = ConfigParser()
# config.read('./config/config.ini')

BATCH_SIZE = 32
BUFFER_SIZE = 1000
IMAGE_DIMENSIONS = (128, 128)

# PATH_TO_TRAIN_DIRECTORY = 'data\\Official Train\\Final_Training\\Images'
# PATH_TO_TRAIN_DIRECTORY = 'data\\One Class Test\\Final Training\\Images'
# training_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_TRAIN_DIRECTORY))

# x_train = tf.keras.utils.image_dataset_from_directory(training_path, batch_size=BATCH_SIZE, image_size=IMAGE_DIMENSIONS,
#                                                      labels=None)
# x_train_processed = ...
# x_train_list = next(iter(x_train))  # convert to list

# PATH_TO_PICTOGRAMS = 'data\\Pictograms'
# pictograms_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_PICTOGRAMS))

# x_pictograms = tf.keras.utils.image_dataset_from_directory(pictograms_path, batch_size=BATCH_SIZE,
#                                                            image_size=IMAGE_DIMENSIONS, labels=None)
# x_pictograms_list = next(iter(x_pictograms))

# generator = make_generator_model()
# generated_images = generator(x_pictograms_list, training=False)
# plt.imshow(generated_images[0])
# plt.show()

# discriminator = make_discriminator_model()
# discriminator_input = generated_images
# guess = discriminator(x_train_list, training=False)

cycle_gan = model.CycleGan()
cycle_gan.compile()

# print(tf.shape(discriminator_input))
# print(tf.shape(guess))
# print(guess)
