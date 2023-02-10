from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import image_augmentation.motion_blur
import image_augmentation.occlusions
import utils.preprocess_image
import model
import toml

config = toml.load('./config/config.toml')
cycle_gan = model.CycleGan(config)

# load image
# pictogram = tf.keras.utils.load_img(r'C:\Users\Frederik\Documents\Studienarbeit\data\Pictograms One Class\138-20.jpg')
pictogram = tf.keras.utils.load_img(r'C:\Users\Frederik\Documents\Studienarbeit\data\Official Train\Final_Training\Images\00005\00058_00029.png')
pictogram = tf.image.resize(pictogram, [256, 256])
img_tensor = tf.keras.utils.img_to_array(pictogram)
img_tensor_batch = tf.expand_dims(img_tensor, 0)

plt.imshow(img_tensor_batch[0, :, :, :].numpy().astype('uint8'))

# generate street sign
# img = cycle_gan.generator_g(img_tensor_batch)

# img_transformed = img[0, :]
# img = Image.open('generator_test/febf9315-95ac-4bc0-b81c-dbffc6636a8a.png').convert('RGB')
# img = Image.open('../data/Pictograms/306.jpg').convert('RGB')
# img = tf.convert_to_tensor(img)
# img_transformed = img
#img = tf.cast(img, tf.float32)
# img_with_tape = image_augmentation.occlusions.add_tape(img)
# img_transformed = image_augmentation.occlusions.add_cross(img_transformed)
# img_transformed = tf.convert_to_tensor(img_transformed)
# img_transformed = image_augmentation.motion_blur.apply_motion_blur(img_transformed, 40, image_augmentation.motion_blur.Direction.DIAGONAL)
# img_transformed = image_augmentation.motion_blur.gamma_correction(img_transformed, 1.7)
# img_transformed = image_augmentation.motion_blur.darken(img_transformed, 0.2)
# img_transformed = image_augmentation.motion_blur.over_exposure(img)
# img_transformed = image_augmentation.motion_blur.sharpen(img)
# plt.imshow(img_with_tape)
# plt.imshow(img_with_cross)
# plt.imshow(img_transformed.numpy())
plt.show()
# img_with_cross = tf.keras.utils.array_to_img(img_with_cross)
# img_with_cross.save(r'{}\{}.png'.format(r"C:\Users\Frederik\Documents\Studienarbeit\data\Pictograms", "Test"))
