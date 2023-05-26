import random
import numpy as np
from enum import Enum
import tensorflow as tf
import cv2
from . import preprocess_image
import tensorflow_addons as tfa
from PIL import Image


class Direction(Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    DIAGONAL = 3


# region Motion Blur
def apply_random_motion_blur(img_tensor):
    """ Applies random motion blur to an image tensor

    Args:
        img_tensor: 3d image tensor that will be blurred (height, width, channels)

    Returns:
        image tensor with applied motion blur
    """
    intensity = np.random.randint(0, 100)
    direction = random.choice(list(Direction))
    return apply_motion_blur(img_tensor, intensity, direction)


def apply_motion_blur(img_tensor, intensity, direction=Direction.HORIZONTAL, min_kernel_size=1, max_kernel_size=30):
    """ Applies motion blur to an image tensor

    Args:
        img_tensor: 3d image tensor that will be blurred (height, width, channels)
        intensity: intensity of the blur [0-100]
        direction: direction of the blur [Direction.VERTICAL (1), Direction.HORIZONTAL (2), Direction.DIAGONAL (3)]
        min_kernel_size: minimum kernel size
        max_kernel_size: maximum kernel size

    Returns:
        image tensor with applied motion blur
    """
    if intensity < 0 or intensity > 100:
        raise ValueError('intensity must be between 0 and 100')
    # convert intensity to kernel size
    kernel_size = int((max_kernel_size - min_kernel_size) * intensity / 100) + min_kernel_size
    # compute kernel
    if direction == Direction.VERTICAL:
        kernel = np.zeros((kernel_size, kernel_size))
        # ones on vertical axis
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    elif direction == Direction.HORIZONTAL:
        # ones on horizontal axis
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    elif direction == Direction.DIAGONAL:
        # ones on diagonal axis; unit matrix
        kernel = np.identity(kernel_size)
    else:
        raise ValueError('direction must be one of Direction.VERTICAL (1), Direction.HORIZONTAL (2), '
                         'Direction.DIAGONAL (3)')
    kernel = kernel / kernel_size
    transformed_img = cv2.filter2D(img_tensor.numpy(), -1, kernel)
    return tf.convert_to_tensor(transformed_img)


# endregion

# region Invalid Street Sign

# region snow
def add_snow(img_tensor, snow_intensity, motion_blur_intensity, motion_blur_direction=Direction.DIAGONAL,
             p_snowflake_min=0.02, p_snowflake_max=0.5):
    """Add snow to an image tensor

    Args:
        img_tensor: 3d image tensor on which snow will be added (height, width, 3)
        snow_intensity: intensity of the snow [0-100]
        motion_blur_intensity: intensity of the motion blur [0-100]
        motion_blur_direction: direction of the motion blur
        p_snowflake_min: minimum probability of a snowflake (default value empirically determined)
        p_snowflake_max: maximum probability of a snowflake (default value empirically determined)

    Returns:
        image tensor with added snow
    """

    # probability for a snowflake
    p = p_snowflake_min + (p_snowflake_max - p_snowflake_min) * snow_intensity / 100

    # Generate random white/black pixels with probability p resp. (1-p); one white pixel equals one snowflake
    # Binomial distribution is used for generation; 1 = white pixel, 0 = black pixel
    seed_1 = tf.random.uniform([], minval=0, maxval=999)
    seed_2 = tf.random.uniform([], minval=0, maxval=999)
    random_particles = tf.random.stateless_binomial(shape=(256, 256, 1), seed=[seed_1, seed_2], counts=1, probs=p,
                                                    output_dtype=tf.float32)
    random_particles = random_particles * 2.0 - 1.0  # normalize from range [0, 1] to range [-1, 1]
    # copy the generated random values for red channel to green, blue and alpha channel 
    random_particles = tf.tile(random_particles, [1, 1, 4])
    # blur the snowflakes
    random_particles = tfa.image.gaussian_filter2d(random_particles, filter_shape=(3, 3), sigma=7)
    # Add wind effect
    random_particles = apply_motion_blur(random_particles, motion_blur_intensity,
                                         motion_blur_direction)

    # add alpha channel to img_tensor (if it doesnt already have one)
    if tf.shape(img_tensor)[-1] < 4:
        img_tensor = tf.concat([img_tensor, tf.ones_like(img_tensor[:, :, 0:1])], axis=-1)
    # paste generated snowflake image onto given image tensor
    snowy_image = tf.math.add(img_tensor, random_particles)
    # normalize to range [-1, 1]
    max_val = tf.math.reduce_max(snowy_image)
    min_val = tf.math.reduce_min(snowy_image)
    snowy_image = tf.math.divide(tf.math.subtract(snowy_image, min_val), tf.math.subtract(max_val, min_val)) * 2 - 1
    # remove 4th channel; now is rgb instead of rgba
    snowy_image = snowy_image[:, :, 0:3]
    return snowy_image


# endregion snow


def make_street_sign_invalid(img_tensor, cross_path, content_size=None, transformation_matrix=None):
    """Makes a street sign invalid by drawing a cross over it.
    Transformation of the street sign can be specified by content_size and transformation_matrix.
    The cross will then be transformed in the same way as the provided street sign.

    Args:
        img_tensor: 3d image tensor (height, width, channels)
        cross_path: path to the cross image (see config file)
        content_size: dimensions of the street sign (content_size = height = width)
        transformation_matrix: transformation matrix of the street sign

    Returns:
        3d image tensor with a cross drawn over the street sign
    """
    invalid_cross = tf.keras.utils.load_img(cross_path, color_mode='rgba', target_size=(256, 256))
    invalid_cross = tf.cast(invalid_cross, tf.float32)
    # normalize image values from [0, 255] to [-1, 1]
    invalid_cross = (invalid_cross / 127.5) - 1
    # transform cross to fit the street sign
    if transformation_matrix is not None and content_size is not None:
        invalid_cross = preprocess_image.transform_image(invalid_cross, content_size, transformation_matrix,
                                                         bg_is_white=False)

    # add alpha channel to img_tensor
    img_tensor = tf.concat([img_tensor, tf.ones_like(img_tensor[:, :, 0:1])], axis=-1)
    img_tensor = img_tensor.numpy()
    invalid_cross = invalid_cross.numpy()
    # convert to PIL image; has to be in range [0, 255]
    img = Image.fromarray((img_tensor * 127.5 + 127.5).astype(np.uint8))
    cross = Image.fromarray((invalid_cross * 127.5 + 127.5).astype(np.uint8))
    # paste cross with transparent background onto img
    img.paste(cross, (0, 0), cross)
    img = tf.convert_to_tensor(np.array(img))
    img = tf.cast(img, tf.float32)
    # normalize image values from [0, 255] to [-1, 1]
    img = (img / 127.5) - 1
    return img
# endregion
