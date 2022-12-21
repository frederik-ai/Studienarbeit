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


def normalize_image_to_255(np_array):
    return (np_array * 127.5) + 1


def shrink_content(image, target_size, content_size):
    image = tf.image.resize(image, content_size)  # shrink the content
    image = 1 - image  # invert the image
    # adds black padding around the image to match target_size
    image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])
    image = 1 - image  # invert back to original color; now padding is white
    return image
