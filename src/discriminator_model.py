import tensorflow as tf
import tensorflow_addons as tfa

KERNEL_SIZE = (4, 4)
INITIAL_FILTER_COUNT = 32
STRIDES = (4, 4)  ###
PADDING = 'same'
LEAK_FACTOR = 0.01


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=INITIAL_FILTER_COUNT, kernel_size=KERNEL_SIZE, strides=STRIDES,
                                     padding=PADDING))
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAK_FACTOR))

    model.add(tf.keras.layers.Conv2D(filters=(INITIAL_FILTER_COUNT * 2), kernel_size=KERNEL_SIZE, strides=STRIDES,
                                     padding=PADDING))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAK_FACTOR))

    model.add(tf.keras.layers.Conv2D(filters=(INITIAL_FILTER_COUNT * 4), kernel_size=KERNEL_SIZE, strides=STRIDES,
                                     padding=PADDING))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAK_FACTOR))

    model.add(tf.keras.layers.Conv2D(filters=(INITIAL_FILTER_COUNT * 8), kernel_size=KERNEL_SIZE, strides=(2,2),
                                     padding=PADDING))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAK_FACTOR))

    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=KERNEL_SIZE, padding=PADDING))

    return model