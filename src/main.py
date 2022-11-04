import tensorflow as tf
import os

# Retrieve Dataset
training_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..',
                                              'data\\Official Train\\Final_Training\\Images'))
x_train = tf.keras.utils.image_dataset_from_directory(training_path, batch_size=32)

# Process Dataset
# ...

# Initialize Parameters
# ...

# region GAN D
# ...
# endregion

# region GAN F
# ...
# endregion

# Define Training Functions
# ...

# Execute Training
# ...
