import tensorflow as tf
import toml
import uuid
import argparse
import math

import utils.load_data
import utils.preprocess_image
import utils.misc
import utils.image_augmentation
import model

# Setup
config = toml.load('./config/config.toml')
BATCH_SIZE = config['training']['batch_size']
IMAGE_SIZE = config['model']['image_size']
config['model']['generator_type'] = 'unet'  # REMOVE

MOTION_BLUR = False
NUM_GENERATED_IMAGES = 5

PATH_TO_PICTOGRAMS = r'C:\Users\Frederik\Documents\Studienarbeit\data\Pictograms One Class'
x_pictograms = tf.keras.utils.image_dataset_from_directory(PATH_TO_PICTOGRAMS, batch_size=BATCH_SIZE,
                                                           image_size=(IMAGE_SIZE, IMAGE_SIZE), labels=None)
x_pictograms_processed = utils.load_data.normalize_dataset(x_pictograms)

cycle_gan = model.CycleGan(config)
cycle_gan.compile()
cycle_gan.restore_latest_checkpoint_if_exists()

# Generate images
num_iterations = math.ceil(NUM_GENERATED_IMAGES / BATCH_SIZE)
num_generated_images_left = NUM_GENERATED_IMAGES
for iteration in range(num_iterations):
    transformed_pictograms = x_pictograms_processed.map(
        lambda t: tf.py_function(utils.preprocess_image.randomly_transform_image_batch, inp=[t],
                                 Tout=tf.float32)
    )
    transformed_pictograms.shuffle(buffer_size=50, reshuffle_each_iteration=True)
    single_pictogram_batch = transformed_pictograms.take(1).get_single_element()
    generator_result = cycle_gan.generator_g(single_pictogram_batch)

    if MOTION_BLUR:
        generator_result = tf.map_fn(lambda t: utils.image_augmentation.apply_motion_blur(t, 100), generator_result)

    if BATCH_SIZE < num_generated_images_left:
        num_images = BATCH_SIZE
        num_generated_images_left -= BATCH_SIZE
    else:
        num_images = num_generated_images_left
    for i in range(num_images):
        utils.misc.store_tensor_as_img(generator_result[i, :], str(uuid.uuid4()), 'generator_test')
