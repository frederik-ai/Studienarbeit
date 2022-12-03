import tensorflow as tf
import tensorflow_addons as tfa

CONV_KERNEL_SIZE = [7, 7]
DOWN_KERNEL_SIZE = [3, 3]
RES_KERNEL_SIZE = [3, 3]
UP_KERNEL_SIZE = [3, 3]
INITIAL_FILTER_COUNT = 32


def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(32))
    
    # Downsample
    model.add(tf.keras.layers.Conv2D(filters=INITIAL_FILTER_COUNT, kernel_size=CONV_KERNEL_SIZE, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Conv2D(filters=INITIAL_FILTER_COUNT * 2, kernel_size=DOWN_KERNEL_SIZE, strides=(2, 2), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Conv2D(filters=INITIAL_FILTER_COUNT * 4, kernel_size=DOWN_KERNEL_SIZE, strides=(2, 2), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    # Residual blocks
    
    
    # Upsample
    model.add(tf.keras.layers.Conv2DTranspose(filters=INITIAL_FILTER_COUNT * 2, kernel_size=UP_KERNEL_SIZE, strides=(2, 2), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=INITIAL_FILTER_COUNT, kernel_size=UP_KERNEL_SIZE, strides=(2, 2), padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=CONV_KERNEL_SIZE, padding='same', activation='tanh'))

    return model
