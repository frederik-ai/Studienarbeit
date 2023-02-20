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
    return


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
# invalid_cross_path = r'C:\Users\Frederik\Documents\Studienarbeit\data\Augmentation\Schild Durchgestrichen.png'


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
