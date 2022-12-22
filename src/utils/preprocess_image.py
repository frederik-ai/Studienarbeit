import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


# taken from https://www.tensorflow.org/tutorials/generative/cyclegan
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # probably needs to be changed if image dimensions change
    return image


def normalize_tensor_of_images(tensor_of_images):
    tensor_of_images = tf.cast(tensor_of_images, tf.float32)
    normalized_tensor = tf.divide(
        tf.subtract(
            tensor_of_images,
            tf.reduce_min(tensor_of_images)
        ),
        tf.subtract(
            tf.reduce_max(tensor_of_images),
            tf.reduce_min(tensor_of_images)
        )
    )


def normalize_image_to_255(np_array):
    return (np_array * 127.5) + 1


def randomly_transform_4d_tensor(image_as_4d_tensor, target_size=(256, 256), max_theta_xy=0.1, max_theta_z=0.0001):
    results = []
    for image in image_as_4d_tensor:
        curr_result = randomly_transform_image(image, target_size, max_theta_xy, max_theta_z)
        results.append(np.expand_dims(curr_result, axis=0))
    return np.concatenate(results, axis=0)


def randomly_transform_image(image, target_size, max_theta_xy=0.1, max_theta_z=0.0001):
    transformed_image = 1 - image  # invert the image

    # randomly shrink the content, add padding to preserve the target_size of the image
    assert (len(target_size) == 2)
    assert (target_size[0] == target_size[1])
    shrink_length = np.random.randint(low=target_size[0] / 4, high=target_size[0])
    shrink_size = (shrink_length, shrink_length)
    transformed_image = tf.image.resize(transformed_image, shrink_size)
    transformed_image = tf.image.resize_with_crop_or_pad(transformed_image, target_size[0], target_size[1])

    # randomly rotate the image in x,y and z direction
    theta_xy = np.random.uniform(low=-max_theta_xy, high=max_theta_xy, size=(1,))[0]  # angle for x,y rotation
    theta_z = np.random.uniform(low=-max_theta_z, high=max_theta_z, size=(1,))[0]  # angle for z rotation
    transformed_image = apply_3d_rotation(transformed_image, theta_xy, theta_z, target_size)

    transformed_image = 1 - transformed_image  # invert the image back to original colors; now black padding is white

    return transformed_image


def apply_3d_rotation(image, theta_xy, theta_z, image_size):
    transformation_matrix = np.array([
        [np.cos(theta_xy), -np.sin(theta_xy), 0],
        [np.sin(theta_xy), np.cos(theta_xy), 0],
        [0, np.sin(theta_z), 1]
    ])
    # image = tf.keras.utils.img_to_array(image)
    rotated_image = cv2.warpPerspective(image.numpy(), transformation_matrix, image_size)
    # rotated_image = tf.keras.utils.array_to_img(rotated_image)
    return rotated_image
