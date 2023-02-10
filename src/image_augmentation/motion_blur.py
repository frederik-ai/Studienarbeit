import os.path
import sys
import numpy as np
import cv2
from enum import Enum

sys.path.insert(0, os.path.realpath('../utils'))
import utils.preprocess_image
import src.model as model
import tensorflow as tf
import tensorflow_addons as tfa
import cv2


class Direction(Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    DIAGONAL = 3


def apply_random_motion_blur(img_tensor):
    return


def apply_motion_blur(img_tensor, intensity, direction=Direction.HORIZONTAL, min_kernel_size=1, max_kernel_size=30):
    """ Applies motion blur to an image tensor

    Args:
        img_tensor: image tensor that will be blurred
        intensity: intensity of the blur (0-100)
        direction: direction of the blur (vertical (1), horizontal (2), diagonal (3))
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


def sharpen(img_tensor):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    transformed_img = cv2.filter2D(img_tensor.numpy(), -1, kernel)
    return transformed_img


# def over_exposure(img_tensor):
#    return tf.math.multiply(1.0, tf.cast(img_tensor, tf.dtypes.float32))

def gamma_correction(img_tensor, gamma, c=255):
    img_tensor = tf.image.adjust_gamma(img_tensor, gamma)
    return tf.image.adjust_gamma(img_tensor, gamma)

def darken(img_tensor, intensitiy):
    return tf.image.adjust_brightness(img_tensor, -intensitiy)
