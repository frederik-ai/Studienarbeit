"""
Utility functions for processing image tensors.

Used for augmentation of street sign pictograms before feeding them to generator 'G'.
Also used for post-processing of generated images.
"""
import tensorflow as tf
import numpy as np
import tensorflow_graphics.image.transformer as tfg_image_transformer


def randomly_transform_image_batch(img_tensor_batch, target_size=256):
    batch_size = img_tensor_batch.shape[0]

    # resize content
    min_content_size = target_size / 1.5
    # we have to work with default python lists, because we need the pop function
    content_sizes = [np.random.randint(low=min_content_size, high=target_size) for el in range(batch_size)]
    content_sizes_tmp = content_sizes[:]  # copy of content_sizes; will be used to pop the elements
    transformed_imgs = tf.map_fn(lambda img: resize_content_of_img(img, target_size, content_sizes_tmp.pop(0)),
                                 img_tensor_batch)

    # randomly rotate the image in x,y and z direction; scale values are empirically chosen
    alpha_z_values = np.random.normal(loc=0.0, scale=3.5, size=batch_size)
    alpha_y_values = np.random.normal(loc=0.0, scale=0.01, size=batch_size)
    alpha_x_values = np.random.normal(loc=0.0, scale=0.01, size=batch_size)
    transform_matrices = np.zeros((batch_size, 3, 3))
    for i in range(batch_size):
        transform_matrices[i] = create_rotation_matrix(alpha_z_values[i], alpha_y_values[i], alpha_x_values[i])
    transform_matrices = tf.convert_to_tensor(transform_matrices, dtype=tf.float32)

    transformed_imgs = 1 - transformed_imgs
    transformed_imgs = tfg_image_transformer.perspective_transform(transformed_imgs, transform_matrices)
    transformed_imgs = 1 - transformed_imgs

    return transformed_imgs, content_sizes, transform_matrices


def transform_image(img_tensor, content_size, transformation_matrix, target_size=256, bg_is_white=True):
    # convert (..., ..., ...) to 4d tensor (1, ..., ..., ...)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    transformation_matrix = tf.expand_dims(transformation_matrix, axis=0)
    # transform
    img_tensor = resize_content_of_img(img_tensor, target_size, content_size, bg_is_white=bg_is_white)
    img_tensor = 1 - img_tensor     # invert the image colors, because padding is black
    img_tensor = tfg_image_transformer.perspective_transform(img_tensor, transformation_matrix,
                                                             border_type=tfg_image_transformer.BorderType.DUPLICATE)
    img_tensor = 1 - img_tensor
    img_tensor = tf.squeeze(img_tensor, axis=0)
    return img_tensor


def create_rotation_matrix(alpha_z, alpha_y, alpha_x, angles_in_degrees=True):
    """Create a rotation matrix from alpha_z, alpha_y and alpha_x angles.
    All angles shall be provided in degrees or radians. Set the parameter 'angles_in_degrees' accordingly.

    Args:
        alpha_z: rotation around z-axis
        alpha_y: rotation around y-axis
        alpha_x: rotation around x-axis
        angles_in_degrees: whether the angles are provided in degrees. Otherwise radians are assumed.

    Returns:
        (3, 3) tensor containing the rotation matrix.
    """
    if angles_in_degrees:
        alpha_z = np.radians(alpha_z)
        alpha_y = np.radians(alpha_y)
        alpha_x = np.radians(alpha_x)

    rotation_matrix_z = np.array([
        [np.cos(alpha_z), -np.sin(alpha_z), 0],
        [np.sin(alpha_z),  np.cos(alpha_z), 0],
        [0,                0,               1]
    ])
    rotation_matrix_y = np.array([
        [np.cos(alpha_y),  0, np.sin(alpha_y)],
        [0,                1,               0],
        [-np.sin(alpha_y), 0, np.cos(alpha_y)]
    ])
    rotation_matrix_x = np.array([
        [1,               0,                0],
        [0, np.cos(alpha_x), -np.sin(alpha_x)],
        [0, np.sin(alpha_x),  np.cos(alpha_x)]
    ])

    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

    return tf.convert_to_tensor(rotation_matrix, dtype=tf.float32)


def resize_content_of_img(img_tensor, target_size, content_size, bg_is_white=True):
    """Resize the content of the image, but keep the original image dimensions.
    A padding is added to the shrunken content. The background of the input image can be white, black or transparent.

    Args:
        img_tensor: 3d tensor of shape (height, width, channels); values normalized to interval [-1, 1]
        target_size: the dimensions of the target image (target_size = width = height)
        content_size: the dimensions of the content of the image (content_size = width = height)
        bg_is_white: whether the background of the image is white

    Returns:
        3d tensor of shape (target_size, target_size, channels)
    """
    if bg_is_white:
        img_tensor = 1 - img_tensor  # invert the image colors, because padding is black
    else:
        img_tensor = (img_tensor + 1) / 2  # convert from range [-1, 1] to [0, 1] because added padding has value 0

    # shrink the content, add padding to preserve the target_size of the image
    shrink_size = (content_size, content_size)
    img_tensor = tf.image.resize(img_tensor, shrink_size)
    img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, target_size, target_size)

    if bg_is_white:
        img_tensor = 1 - img_tensor  # revert to original colors
    else:
        img_tensor = (img_tensor * 2) - 1  # convert back to range [-1, 1]

    return img_tensor
