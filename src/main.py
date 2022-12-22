import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from generator_model import make_generator_model
# from discriminator_model import make_discriminator_model
import utils.load_data
import utils.preprocess_image
# from configparser import ConfigParser
import model

# config = ConfigParser()
# config.read('./config/config.ini')

BATCH_SIZE = 16
BUFFER_SIZE = 1000
IMAGE_DIMENSIONS = (128, 128)

# PATH_TO_TRAIN_DIRECTORY = 'data\\Official Train\\Final_Training\\Images'
# PATH_TO_TRAIN_DIRECTORY = 'data\\Mini One Class Test\\Final Training\\Images'
# PATH_TO_TRAIN_DIRECTORY = 'data\\Official Train\\Final_Training\\Images\\00013'
PATH_TO_TRAIN_DIRECTORY = 'data/Official Train/Final_Training/Images/00013'
training_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_TRAIN_DIRECTORY))

x_train = tf.keras.utils.image_dataset_from_directory(training_path, batch_size=BATCH_SIZE, image_size=IMAGE_DIMENSIONS,
                                                      labels=None)
x_train_processed = utils.load_data.normalize_dataset(x_train)
# x_train_list = next(iter(x_train))  # convert to list
# x_train_list = x_train.get_single_element()

# PATH_TO_PICTOGRAMS = 'data\\Pictograms'
PATH_TO_PICTOGRAMS = 'data/Pictograms'
pictograms_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', PATH_TO_PICTOGRAMS))

x_pictograms = tf.keras.utils.image_dataset_from_directory(pictograms_path, batch_size=BATCH_SIZE,
                                                           image_size=IMAGE_DIMENSIONS, labels=None)

x_pictograms_list = next(iter(x_pictograms))
first_pictogram = x_pictograms_list[0]
# first_pictogram = tf.image.resize(first_pictogram, (128, 128))
# first_pictogram = 1 - first_pictogram # invert
# plt.imshow(first_pictogram)
# plt.show()
# first_pictogram = tf.image.resize_with_crop_or_pad(first_pictogram, 256, 256)
# first_pictogram = 1 - first_pictogram # invert back
# first_pictogram = utils.preprocess_image.shrink_content(first_pictogram, IMAGE_DIMENSIONS, (128, 128))
# first_pictogram = utils.preprocess_image.apply_random_3d_rotation(first_pictogram, IMAGE_DIMENSIONS)
first_pictogram = utils.preprocess_image.randomly_transform_image(first_pictogram, IMAGE_DIMENSIONS)
plt.imshow(first_pictogram)
plt.show()
print('')

# create list that contains the pictogram 10 times
# x_pictograms_list = [next(iter(x_pictograms))[0]] * 10
# x_pictograms_list = next(iter(x_pictograms))
# x_pictograms_list = x_pictograms.get_single_element()

# generator = make_generator_model()
# generated_images = generator(x_pictograms_list, training=False)
# plt.imshow(generated_images[0])
# plt.show()

# discriminator = make_discriminator_model()
# discriminator_input = generated_images
# guess = discriminator(x_train_list, training=False)

cycle_gan = model.CycleGan()
cycle_gan.print_welcome()
cycle_gan.restore_latest_checkpoint_if_exists()
cycle_gan.compile()
# generator_test_result = cycle_gan.generator_g(x_pictograms.get_single_element()) # USE TEST DATA NOT TRAIN DATA
cycle_gan.fit(x_pictograms.get_single_element(), x_train_processed, epochs=100)

# cycle_gan.train_step(x_train_list, x_pictograms_list
# generated_images = cycle_gan.generator_g(x_pictograms.get_single_element(), training=False)

# img = generated_images[0]
# img = img.numpy()
# img = tf.keras.utils.array_to_img(img)
# img_normalized_to_255 = utils.preprocess_image.normalize_image_to_255(img)
# image = Image.fromarray(img).convert("RGB")
# img.save('result.png')
# plt.imshow(generated_images[0])
# plt.show()

# print(tf.shape(discriminator_input))
# print(tf.shape(guess))
# print(guess)
