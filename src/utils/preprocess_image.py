import tensorflow as tf


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
