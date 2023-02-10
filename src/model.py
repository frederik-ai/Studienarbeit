import tensorflow as tf
import time
from tqdm import tqdm
import uuid
import os

from tensorflow_examples.models.pix2pix import pix2pix
import utils.misc
import utils.load_data
import utils.preprocess_image
import resnet


# Partly inspired by: https://www.tensorflow.org/tutorials/generative/cyclegan

class GeneratorType:
    RESNET = 0
    U_NET = 1


class CycleGan:

    def __init__(self, config):
        if config['model']['generator_type'] == 'unet':
            self.generator_type = GeneratorType.U_NET
        elif config['model']['generator_type'] == 'resnet':
            self.generator_type = GeneratorType.RESNET
        self.image_size = config['model']['image_size']

        # Generators
        if self.generator_type == GeneratorType.U_NET:
            self.generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
            self.generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')
        else:
            self.generator_g = resnet.ResnetGenerator((self.image_size, self.image_size, 3), n_blocks=9)
            self.generator_f = resnet.ResnetGenerator((self.image_size, self.image_size, 3), n_blocks=9)
        # Discriminators
        if self.generator_type == GeneratorType.U_NET:
            self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
            self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
        else:
            self.discriminator_x = resnet.ConvDiscriminator((256, 256, 3))
            self.discriminator_y = resnet.ConvDiscriminator((256, 256, 3))

        # Optimizers
        self.generator_g_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

        # Loss functions
        if self.generator_type == GeneratorType.U_NET:
            # Showed better performance with unet
            self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.adversarial_loss = tf.keras.losses.MeanSquaredError()
        self.LAMBDA = config['training']['lambda']

        # checkpoints
        self.checkpoint = self.init_checkpoint()
        if self.generator_type == GeneratorType.U_NET:
            checkpoint_path = './checkpoints/cyclegan_u_net'
        else:
            checkpoint_path = './checkpoints/cyclegan_resnet'
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=3)

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
            for image_batch in tqdm(real_images):
                transformed_pictograms = pictograms.map(
                    lambda t: tf.py_function(utils.preprocess_image.randomly_transform_image_batch, inp=[t],
                                             Tout=tf.float32)
                )
                transformed_pictograms.shuffle(buffer_size=20, reshuffle_each_iteration=True)
                single_pictogram_batch = transformed_pictograms.take(1).get_single_element()
                self.train_step(single_pictogram_batch, image_batch)

            transformed_pictograms = pictograms.map(
                lambda t: tf.py_function(utils.preprocess_image.randomly_transform_image_batch, inp=[t],
                                         Tout=tf.float32)
            )
            transformed_pictograms.shuffle(buffer_size=150, reshuffle_each_iteration=True)
            single_pictogram = transformed_pictograms.take(1).get_single_element()
            generator_test_result = self.generator_g(single_pictogram)
            random_filename = str(uuid.uuid4())
            utils.misc.store_tensor_as_img(tensor=generator_test_result[0], filename=random_filename,
                                           relative_path='generated_images')

            self.checkpoint_manager.save()
            print('Checkpoint saved for epoch {}.'.format(epoch + 1))
            # if ((epoch + 1) % 1 == 0) and ((epoch + 1) < epochs):
            #    self.checkpoint_manager.save()
            #    print('Checkpoint saved for epoch {}.'.format(epoch + 1))

        print('Time taken for training is {:.2f} sec\n'.format(time.time() - start))
        self.checkpoint_manager.save()
        print('Checkpoint saved for this training')

    def discriminator_loss(self, real, generated):
        real_loss = self.adversarial_loss(tf.ones_like(real), real)
        generated_loss = generated_loss = self.adversarial_loss(
            tf.zeros_like(generated), generated)
        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss * 0.5

    def generator_loss(self, generated):
        return self.adversarial_loss(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def train_step(self, real_pictogram, real_street_sign):
        with tf.GradientTape(persistent=True) as tape:
            generated_street_sign = self.generator_g(real_pictogram, training=True)
            cycled_pictogram = self.generator_f(generated_street_sign, training=True)

            generated_pictogram = self.generator_f(real_street_sign, training=True)
            cycled_street_sign = self.generator_g(generated_pictogram, training=True)

            discriminator_x_real_pictogram = self.discriminator_x(real_pictogram, training=True)
            discriminator_y_real_street_sign = self.discriminator_y(real_street_sign, training=True)

            discriminator_x_fake_pictogram = self.discriminator_x(generated_pictogram, training=True)
            discriminator_y_fake_street_sign = self.discriminator_y(generated_street_sign, training=True)

            # calculate the loss
            generator_g_loss = self.generator_loss(discriminator_y_fake_street_sign)
            generator_f_loss = self.generator_loss(discriminator_x_fake_pictogram)

            total_cycle_loss = self.calc_cycle_loss(
                real_pictogram, cycled_pictogram) + self.calc_cycle_loss(real_street_sign, cycled_street_sign)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = generator_g_loss + total_cycle_loss
            total_gen_f_loss = generator_f_loss + total_cycle_loss

            disc_x_loss = self.discriminator_loss(discriminator_x_real_pictogram, discriminator_x_fake_pictogram)
            disc_y_loss = self.discriminator_loss(discriminator_y_real_street_sign, discriminator_y_fake_street_sign)

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
            print("Using the latest checkpoint of CycleGAN.")
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
