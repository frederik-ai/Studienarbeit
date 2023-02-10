import tensorflow as tf
import cv2
import numpy as np


def randomly_transform_image_batch(image_batch, target_size=256, max_theta_xy=0.1, max_theta_z=0.0001):
    image_batch = tf.map_fn(lambda image: randomly_transform_image(image, target_size, max_theta_xy, max_theta_z),
                            image_batch)
    return image_batch


def randomly_transform_image(image, target_size, max_theta_xy=0.1, max_theta_z=0.0001):
    transformed_image = 1 - image  # invert the image

    # randomly shrink the content, add padding to preserve the target_size of the image
    shrink_length = np.random.randint(low=target_size / 2, high=target_size)
    shrink_size = (shrink_length, shrink_length)
    transformed_image = tf.image.resize(transformed_image, shrink_size)
    transformed_image = tf.image.resize_with_crop_or_pad(transformed_image, target_size, target_size)

    # randomly rotate the image in x,y and z direction
    theta_xy = np.random.uniform(low=-max_theta_xy, high=max_theta_xy, size=(1,))[0]  # angle for x,y rotation
    theta_z = np.random.uniform(low=-max_theta_z, high=max_theta_z, size=(1,))[0]  # angle for z rotation
    transformed_image = apply_3d_rotation(transformed_image, theta_xy, theta_z, target_size)

    # invert the image back to original colors; now black padding is white
    transformed_image = 1 - transformed_image

    return transformed_image


def apply_3d_rotation(image, theta_xy, theta_z, image_size):
    transformation_matrix = np.array([
        [np.cos(theta_xy), -np.sin(theta_xy), 0],
        [np.sin(theta_xy), np.cos(theta_xy), 0],
        [0, np.sin(theta_z), 1]
    ])
    rotated_image = cv2.warpPerspective(image.numpy(), transformation_matrix, (image_size, image_size))
    return rotated_image
