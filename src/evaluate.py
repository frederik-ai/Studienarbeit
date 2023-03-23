import toml
import tensorflow as tf
from classifier import model

config = toml.load('config/config.toml')
batch_size = config['training']['batch_size']
image_size = config['model']['image_size']

# paths
test_data_path = config['paths']['test_data']
real_train_data_path = config['paths']['train_data']

classifier = model.create_model()

test_data = tf.keras.utils.image_dataset_from_directory(test_data_path, batch_size=batch_size, label_mode='categorical',
                                                        image_size=(image_size, image_size), shuffle=True)

real_train_data = tf.keras.utils.image_dataset_from_directory(real_train_data_path, batch_size=batch_size,
                                                              label_mode='categorical',
                                                              image_size=(image_size, image_size))

classifier.fit(x=real_train_data, validation_data=test_data, epochs=5)
# Only real data

# Only Unet data

# Only ResNet data

# ResNEt and UNet data

# real data + Unet and ResNet data
