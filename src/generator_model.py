import tensorflow as tf
import tensorflow_addons as tfa


def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.InputLayer(input_shape=x_train_shape))
    # model.add(tf.keras.layers.Dense(32))
    # model.add(tfa.layers.InstanceNormalization())  # add batch normalization before activation of input layer
    # model.add(tf.keras.layers.ReLU())

    # TODO: Change filter size
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(4, 4), padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(4, 4), padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(4, 4), padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(4, 4), padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation='tanh'))
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same'))
    # model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same'))
    # model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu'))
    # model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='tanh'))
    return model