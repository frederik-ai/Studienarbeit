from PIL import Image
import numpy as np
import tensorflow as tf
import os

def store_tensor_as_img(tensor, filename, relative_path=''):
   image = tensor.numpy()
   image = tf.keras.utils.array_to_img(image)
   full_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', relative_path)) # BAD
   image.save(r'{}\{}.png'.format(full_path, filename))