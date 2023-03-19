import tensorflow as tf
import tensorflow_graphics as tfg
import tensorflow_addons as tfa
import utils.misc
import uuid
import matplotlib.pyplot as plt
import utils.image_augmentation
from PIL import Image
import numpy as np


# Populate tensor with random values of -1 or 1
# random_particles = tf.random.uniform(shape=(256, 256, 1), minval=0, maxval=2, dtype=tf.int32) * 2 - 1
# random_particles = tf.tile(random_particles, [1, 1, 3])
# random_particles = tf.cast(random_particles, tf.float32)
# utils.misc.store_tensor_as_img(random_particles, 'random_particles.png', 'generated_images')
# def add_snow(img_tensor, snow_intensity, motion_blur_intensity, motion_blur_direction, p_snowflake_min=0.02, p_snowflake_max=0.5):
#    """Add snow to an image tensor
#
#    Args:
#        img_tensor: 3d image tensor on which snow will be added (height, width, 3)
#        snow_intensity: intensity of the snow [0-100]
#        motion_blur_intensity: intensity of the motion blur [0-100]
#        motion_blur_direction: direction of the motion blur
#        p_snowflake_min: minimum probability of a snowflake (default value empirically determined)
#        p_snowflake_max: maximum probability of a snowflake (default value empirically determined)

#    Returns:
#        image tensor with added snow
#    """

#    # probability for a snowflake
#    p = p_snowflake_min + (p_snowflake_max - p_snowflake_min) * snow_intensity / 100

    # Generate random white/black pixels with probability p/(1-p); one white pixel equals one snowflake
#    random_particles = tf.random.stateless_binomial(shape=(256, 256, 1), seed=[123, 456], counts=1, probs=p,
#                                                    output_dtype=tf.float32)
#    random_particles = random_particles * 2.0 - 1.0  # normalize to range [-1, 1]
#    random_particles = tf.tile(random_particles, [1, 1, 4])  # add rgba channel
#    random_particles = tfa.image.gaussian_filter2d(random_particles, filter_shape=(3, 3), sigma=7)  # blur the snowflakes
    # Add wind effect
#    random_particles = utils.image_augmentation.apply_motion_blur(random_particles, motion_blur_intensity, motion_blur_direction)

    # add alpha channel to img_tensor
#    img_tensor = tf.concat([img_tensor, tf.ones_like(img_tensor[:, :, 0:1])], axis=-1)
    # paste generated snowflake image onto given image tensor
#    ret_image = tf.math.add(img_tensor, random_particles)
    # remove 4th channel; now is rgb instead of rgba
#    ret_image = ret_image[:, :, 0:3]
#    return ret_image


#def main():
#    bg_image = tf.keras.utils.load_img(
#        r'C:\Users\Frederik\Documents\Studienarbeit\src\generated_images\21932ea0-767c-41b3-abe3-1943f7a31683.png',
#        color_mode='rgb')
#    bg_image = tf.keras.preprocessing.image.img_to_array(bg_image)
#    bg_image = utils.misc.normalize_img(bg_image)
#    ret_image = add_snow(bg_image, 20, 15, utils.image_augmentation.Direction.DIAGONAL)
#    utils.misc.store_tensor_as_img(ret_image, str(uuid.uuid4()), 'generated_images')


#if __name__ == '__main__':
#    main()
