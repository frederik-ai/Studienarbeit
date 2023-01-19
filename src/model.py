import tensorflow as tf
import time
from tqdm import tqdm
import uuid
import os
from tensorflow_examples.models.pix2pix import pix2pix
import utils.misc
import utils.load_data
import utils.preprocess_image

# Inspired by: https://www.tensorflow.org/tutorials/generative/cyclegan

OUTPUT_CHANNELS = 3


class CycleGan:

    def __init__(self, image_size, batch_size):
        # GAN 'G'
        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        # GAN 'F'
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        # optimizers
        self.generator_g_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5, beta_2=0.999)
        self.generator_f_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5, beta_2=0.999)
        self.discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5, beta_2=0.999)
        self.discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5, beta_2=0.999)

        # for computation of loss
        self.adversarial_loss_function = tf.keras.losses.MeanSquaredError()
        self.l1_loss_function = tf.keras.losses.MeanAbsoluteError()
        self.LAMBDA = 10

        # checkpoints
        self.checkpoint = self.init_checkpoint()
        if os.name == 'posix':
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './checkpoints', max_to_keep=3)
        elif os.name == 'nt':
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './checkpoints', max_to_keep=3)

        self.image_size = image_size
        self.batch_size = batch_size

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
                transformed_pictograms = pictograms.map(
                    lambda t: tf.py_function(utils.preprocess_image.randomly_transform_4d_tensor, inp=[t],
                                             Tout=tf.float32)
                )
                transformed_pictograms = pictograms.map(
                    lambda t: tf.py_function(utils.preprocess_image.randomly_transform_4d_tensor, inp=[t],
                                             Tout=tf.float32)
                    )
                transformed_pictograms.shuffle(buffer_size=150, reshuffle_each_iteration=True)
                multiple_pictograms = transformed_pictograms.take(self.batch_size)
                # this will only be executed once; multiple_pictograms contains batch_size elements
                for pictogram_batch in multiple_pictograms:
                    self.train_step(pictogram_batch, image)

            transformed_pictograms = pictograms.map(
                lambda t: tf.py_function(utils.preprocess_image.randomly_transform_4d_tensor, inp=[t],
                                         Tout=tf.float32)
            )
            transformed_pictograms.shuffle(buffer_size=150, reshuffle_each_iteration=True)
            single_pictogram = transformed_pictograms.take(1).get_single_element()
            generator_test_result = self.generator_g(single_pictogram)
            random_filename = str(uuid.uuid4())
            utils.misc.store_tensor_as_img(tensor=generator_test_result[0], filename=random_filename,
                                           relative_path='generated_images')

            if ((epoch + 1) % 1 == 0) and ((epoch + 1) < epochs):
                self.checkpoint_manager.save()
                print('Checkpoint saved for epoch {}.'.format(epoch + 1))

        print('Time taken for training is {:.2f} sec\n'.format(time.time() - start))
        self.checkpoint_manager.save()
        print('Checkpoint saved for this training')

    def train_step(self, real_pictogram, real_street_sign):
        with tf.GradientTape(persistent=True) as tape:
            fake_street_sign = self.generator_g(real_pictogram, training=True)
            fake_pictogram = self.generator_f(real_street_sign, training=True)
            discriminator_x_guess_fake = self.discriminator_x(fake_pictogram, training=True)
            discriminator_x_guess_real = self.discriminator_x(real_pictogram, training=True)
            discriminator_y_guess_fake = self.discriminator_y(fake_street_sign, training=True)
            discriminator_y_guess_real = self.discriminator_y(real_street_sign, training=True)

            # Discriminator X Adversarial Loss
            # -- How well does discriminator x detect fake street signs?
            ground_truth_real = tf.ones_like(discriminator_x_guess_real)
            discriminator_x_real_loss = self.adversarial_loss_function(ground_truth_real, discriminator_x_guess_real)
            ground_truth_fake = tf.zeros_like(discriminator_x_guess_fake)
            discriminator_x_fake_loss = self.adversarial_loss_function(ground_truth_fake, discriminator_x_guess_fake)
            discriminator_x_loss = discriminator_x_real_loss + discriminator_x_fake_loss

            # Discriminator Y Adversarial Loss
            # -- How well does discriminator y detect fake pictograms?
            ground_truth_real = tf.ones_like(discriminator_y_guess_real)
            discriminator_y_real_loss = self.adversarial_loss_function(ground_truth_real, discriminator_y_guess_real)
            ground_truth_fake = tf.zeros_like(discriminator_y_guess_fake)
            discriminator_y_fake_loss = self.adversarial_loss_function(ground_truth_fake, discriminator_y_guess_fake)
            discriminator_y_loss = discriminator_y_real_loss + discriminator_y_fake_loss

            # Generators Adversarial Loss
            # -- How well do the generators fool the discriminators?
            inverse_ground_truth_fake = tf.ones_like(discriminator_y_guess_fake)
            generator_g_adv_loss = self.adversarial_loss_function(inverse_ground_truth_fake, discriminator_y_guess_fake)
            inverse_ground_truth_fake = tf.ones_like(discriminator_x_guess_fake)
            generator_f_adv_loss = self.adversarial_loss_function(inverse_ground_truth_fake, discriminator_x_guess_fake)

            # Cycle Loss
            # -- When taking a pictogram, generating a street sign image from it
            # -- and then generating it back to a pictogram: How similar are the two pictograms?
            cycled_pictogram = self.generator_f(fake_street_sign, training=True)
            cycled_pictogram_loss = self.l1_loss_function(real_pictogram, cycled_pictogram)
            cycled_street_sign = self.generator_g(fake_pictogram, training=True)
            cycled_street_sign_loss = self.l1_loss_function(real_street_sign, cycled_street_sign)
            total_cycle_loss = cycled_pictogram_loss + cycled_street_sign_loss

            # Generators Total Loss
            # Cycle consistency loss is multiplied by the weight lambda to make it more important
            # This means it is very important to us that the generated street sign contains the actual pictogram
            generator_g_loss = generator_g_adv_loss + total_cycle_loss * self.LAMBDA
            generator_f_loss = generator_f_adv_loss + total_cycle_loss * self.LAMBDA

        generator_g_gradients = tape.gradient(generator_g_loss, self.generator_g.trainable_variables)
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))
        generator_f_gradients = tape.gradient(generator_f_loss, self.generator_f.trainable_variables)
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))
        discriminator_x_gradients = tape.gradient(discriminator_x_loss, self.discriminator_x.trainable_variables)
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))
        discriminator_y_gradients = tape.gradient(discriminator_y_loss, self.discriminator_y.trainable_variables)
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
