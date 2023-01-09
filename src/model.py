import tensorflow as tf
import time
from tqdm import tqdm
import uuid
import os

# import generator_model
# import discriminator_model
from tensorflow_examples.models.pix2pix import pix2pix
import utils.misc
import utils.load_data
import utils.preprocess_image

# A lot of code taken from: https://www.tensorflow.org/tutorials/generative/cyclegan

OUTPUT_CHANNELS = 3


class CycleGan:

    def __init__(self, image_size):
        # GAN 'G'
        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        # GAN 'F'
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        # optimizers
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # for computation of loss
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = 10

        # checkpoints
        self.checkpoint = self.init_checkpoint()
        if os.name == 'posix':
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './checkpoints', max_to_keep=1)
        elif os.name == 'nt':
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './src/checkpoints', max_to_keep=1)

        self.image_size = image_size

    def compile(self):
        self.generator_g.compile()
        self.discriminator_x.compile()
        self.generator_f.compile()
        self.discriminator_y.compile()

    def generate(self, input_pictogram):
        return self.generator_g(input_pictogram, training=False)

    def fit(self, pictograms, real_images, epochs=1):
        print('Training...')

        start = time.time()
        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1} / {epochs}')
            for image in tqdm(real_images):
                transf_pictos = pictograms.map(lambda t: tf.py_function(utils.preprocess_image.randomly_transform_4d_tensor,
                                                                        inp=[t],
                                                                        Tout=tf.float32))
                single_pictogram = transf_pictos.get_single_element()
                self.train_step(single_pictogram, image)

            generator_test_result = self.generator_g(pictograms.get_single_element())  # USE TEST DATA NOT TRAIN DATA
            random_filename = str(uuid.uuid4())
            utils.misc.store_tensor_as_img(tensor=generator_test_result[0], filename=random_filename,
                                           relative_path='generated_images')

            if ((epoch + 1) % 50 == 0) and ((epoch + 1) < epochs):
                self.checkpoint_manager.save()
                print('Checkpoint saved for epoch {}.'.format(epoch + 1))

        print('Time taken for training is {:.2f} sec\n'.format(time.time() - start))
        self.checkpoint_manager.save()
        print('Checkpoint saved for this training')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = generated_loss = self.loss_obj(
            tf.zeros_like(generated), generated)
        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(
                real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + \
                               self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + \
                               self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

    def init_checkpoint(self):
        checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                                         generator_f=self.generator_f,
                                         discriminator_x=self.discriminator_x,
                                         discriminator_y=self.discriminator_y,
                                         generator_g_optimizer=self.generator_g_optimizer,
                                         generator_f_optimizer=self.generator_f_optimizer,
                                         discriminator_x_optimizer=self.discriminator_x_optimizer,
                                         discriminator_y_optimizer=self.discriminator_y_optimizer)
        return checkpoint

    def restore_latest_checkpoint_if_exists(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Using the latest chekckpoint of CycleGAN.")
        else:
            print("No checkpoint found for CycleGAN. Starting from scratch.")

    def print_welcome(self):
        print("""
   ___        _      ___   _   _  _ 
  / __|  _ __| |___ / __| /_\ | \| |
 | (_| || / _| / -_) (_ |/ _ \| .` |
  \___\_, \__|_\___|\___/_/ \_\_|\_|
      |__/                                              
            """)
