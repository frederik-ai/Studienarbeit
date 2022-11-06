import tensorflow as tf
import os

BATCH_SIZE = 32
IMAGE_DIMENSIONS = (256, 256)

# Retrieve Dataset
training_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..',
                                              'data\\Official Train\\Final_Training\\Images'))

# x_train is a data.Dataset object
x_train = tf.keras.utils.image_dataset_from_directory(training_path, batch_size=BATCH_SIZE, image_size=IMAGE_DIMENSIONS)
x_train_shape = tf.shape(x_train)


# GAN
# # Generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=x_train_shape))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.BatchNormalization())  # add batch normalization before activation of input layer
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                                     input_shape=x_train_shape))
    # TODO: add fractional-strided convolution instead of pooling
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='tanh',
                                     input_shape=x_train_shape))
    return model


# # Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    # TODO: add layers - use LeakyReLu in all layers

    return model


# %% Generate
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define Training Functions
# ...

# Execute Training
# ...
