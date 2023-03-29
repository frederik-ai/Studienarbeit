"""
Implementation of the CycleGAN model. Partly derived from: https://www.tensorflow.org/tutorials/generative/cyclegan
"""

import tensorflow as tf
import time
import pickle
from tqdm import tqdm
import uuid
import os
from enum import Enum

from tensorflow_examples.models.pix2pix import pix2pix
import utils.misc
import utils.load_data
import utils.preprocess_image
import resnet


class GeneratorType(Enum):
    RESNET = 0
    U_NET = 1


class CycleGan:
    """Implementation of the CycleGan model.

    Args:
        config (dict): The configuration of the model.

    Attributes:
        generator_type (GeneratorType): Type of generator model that will be used.
        image_size (int): Dimensions of the generated images (image_size = width = height).
        generator_g: Generator that translates from pictogram to street sign image.
        generator_f: Generator that translates from street sign image to pictogram.
        discriminator_x: Discriminator that discriminates between real and generated street sign images.
        discriminator_y: Discriminator that discriminates between real and generated pictograms.
        generator_g_optimizer: Optimizer for generator_g.
        generator_f_optimizer: Optimizer for generator_f.
        discriminator_x_optimizer: Optimizer for discriminator_x.
        discriminator_y_optimizer: Optimizer for discriminator_y.
        adversarial_loss: Loss function for the adversarial loss.
        l1_loss: L1 loss function.
        LAMBDA (int): Weight of the cycle-consistency loss compared to adversarial loss.
        checkpoint: Current checkpoint for the generators and the discriminators.
        checkpoint_manager: Loads, stores and deletes checkpoints.
    """

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
        self.l1_loss = tf.keras.losses.MeanAbsoluteError()
        self.LAMBDA = config['training']['lambda']
        
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # For Tensorboard
        self.total_epochs = tf.Variable(0)
        self.total_steps = tf.Variable(0)
        self.log_path = 'logs/' + config['model']['generator_type']
        self.summary_writer = tf.summary.create_file_writer(self.log_path)

        # Checkpoints
        self.checkpoint = self.init_checkpoint()
        if self.generator_type == GeneratorType.U_NET:
            checkpoint_path = './checkpoints/cyclegan_u_net'
        else:
            checkpoint_path = './checkpoints/cyclegan_resnet'
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=3)

    def compile(self):
        """Compile all generators and discriminators."""
        self.generator_g.compile()
        self.discriminator_x.compile()
        self.generator_f.compile()
        self.discriminator_y.compile()

    def generate(self, input_pictogram):
        """Generate a batch of street sign images from a batch of pictograms.

        Args:
            input_pictogram: 4d tensor containing the ([transformed](/utils/preprocess_image.html)) pictograms
            (batch_size, height, width, channels).

        Returns:
            4d tensor containing the generated street sign images (batch_size, height, width, channels).
        """
        return self.generator_g(input_pictogram, training=False)

    def fit(self, pictograms, real_images, epochs=1):
        """Train the model for a specific number of epochs. Checkpoints are saved every epoch.

        Args:
            pictograms: 4d tensor containing the raw pictograms (batch_size, height, width, channels).
            real_images: 4d tensor containing the training images of street signs (batch_size, height, width, channels).
            epochs: Number of epochs to train the model.
        """
        print('Training...')
        with self.summary_writer.as_default():
            for epoch in range(epochs):
                self.total_epochs.assign_add(1)  # increment
                print(f'Epoch: {int(self.total_epochs)} / {int(self.total_epochs) + epochs-(epoch+1)}')

                # Single training step
                for image_batch in tqdm(real_images):
                    self.total_steps.assign_add(1)  # increment

                    # Transform the pictograms
                    pictograms.shuffle(buffer_size=100, reshuffle_each_iteration=True)
                    single_pictogram_batch = pictograms.take(1).get_single_element()
                    single_pictogram_batch, _, _ = utils.preprocess_image.randomly_transform_image_batch(
                        single_pictogram_batch)

                    # Train the model
                    losses = self.train_step(single_pictogram_batch, image_batch)

                    # For Tensorboard; Log the losses
                    for loss_name in losses:
                        tf.summary.scalar(loss_name, losses[loss_name], int(self.total_steps))

                # After each epoch: Generate an image
                pictograms.shuffle(buffer_size=100, reshuffle_each_iteration=True)
                single_pictogram_batch = pictograms.take(1).get_single_element()
                single_pictogram_batch, _, _ = utils.preprocess_image.randomly_transform_image_batch(
                    single_pictogram_batch)
                generator_test_result = self.generator_g(single_pictogram_batch)
                random_filename = str(uuid.uuid4())
                utils.misc.store_tensor_as_img(tensor=generator_test_result[0], filename=random_filename,
                                               relative_path='generated_images')
                
                self.summary_writer.flush()  # write tensorboard logs to log file

                if ((epoch + 1) % 10 == 0) and ((epoch + 1) < epochs):
                     self.checkpoint_manager.save()
                     print('Checkpoint saved for epoch {}.'.format(epoch + 1))
        self.checkpoint_manager.save()
        print('Checkpoint saved for this training')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = generated_loss = self.loss_obj(
            tf.zeros_like(generated), generated)
        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss * 0.5

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def train_step(self, real_x, real_y):
        """Train the model on a single batch of images.

        Args:
            real_pictogram: 4d tensor containing the pictograms (batch_size, height, width, channels).
            real_street_sign: 4d tensor containing real street sign images (batch_size, height, width, channels).

        Returns:
            Losses (dict): Dictionary containing the losses of the generators and discriminators.
        """
        with tf.GradientTape(persistent=True) as tape:
            real_pictogram = real_x
            real_street_sign = real_y
            
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)
            
            # Generate
            generated_street_sign = self.generator_g(real_pictogram, training=True)
            generated_pictogram = self.generator_f(real_street_sign, training=True)

            # Discriminator X loss
            # -- how well does it identify real pictograms as real?
            discriminator_x_real_pictogram = self.discriminator_x(real_pictogram, training=True)
            discriminator_x_real_loss = self.adversarial_loss(tf.ones_like(discriminator_x_real_pictogram),
                                                              discriminator_x_real_pictogram)
            # -- how well does it identify fake pictograms as fake?
            discriminator_x_fake_pictogram = self.discriminator_x(generated_pictogram, training=True)
            discriminator_x_fake_loss = self.adversarial_loss(tf.zeros_like(discriminator_x_fake_pictogram),
                                                              discriminator_x_fake_pictogram)
            # -- total loss
            disc_x_loss = (discriminator_x_real_loss + discriminator_x_fake_loss) * 0.5

            # Discriminator Y loss
            # -- how well does it identify real traffic signs as real?
            discriminator_y_real_street_sign = self.discriminator_y(real_street_sign, training=True)
            discriminator_y_real_loss = self.adversarial_loss(tf.ones_like(discriminator_y_real_street_sign),
                                                              discriminator_y_real_street_sign)
            # -- how well does it identify fake traffic signs as fake?
            discriminator_y_fake_street_sign = self.discriminator_y(generated_street_sign, training=True)
            discriminator_y_fake_loss = self.adversarial_loss(tf.zeros_like(discriminator_y_fake_street_sign),
                                                              discriminator_y_fake_street_sign)
            # -- total loss
            disc_y_loss = (discriminator_y_real_loss + discriminator_y_fake_loss) * 0.5

            # Cycle consistency loss
            # -- pictogram -> street sign -> pictogram
            # -- how well does the input pictogram match the generated pictogram?
            cycled_pictogram = self.generator_f(generated_street_sign, training=True)
            cycled_street_sign = self.generator_g(generated_pictogram, training=True)
            total_cycle_loss = self.calc_cycle_loss(
                real_pictogram, cycled_pictogram) + self.calc_cycle_loss(real_street_sign, cycled_street_sign)

            # Generators G&F loss
            # -- how well do the generators fool the discriminators?
            generator_g_loss = self.adversarial_loss(tf.ones_like(discriminator_y_fake_street_sign),
                                                     discriminator_y_fake_street_sign)
            generator_f_loss = self.adversarial_loss(tf.ones_like(discriminator_x_fake_pictogram),
                                                     discriminator_x_fake_pictogram)
            total_gen_g_loss = generator_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = generator_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            total_loss = disc_x_loss + disc_y_loss + total_gen_g_loss + total_gen_f_loss

        # Gradients for optimization
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        return {
            'discriminator_x_loss': disc_x_loss,
            'discriminator_y_loss': disc_y_loss,
            'generator_g_loss': total_gen_g_loss,
            'generator_f_loss': total_gen_f_loss,
            'total_loss': total_loss
        }

    def init_checkpoint(self):
        """Create a checkpoint object.

        Returns:
            A Tensorflow [checkpoint object](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)
            for all the components of the model.
        """
        checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                                         generator_f=self.generator_f,
                                         discriminator_x=self.discriminator_x,
                                         discriminator_y=self.discriminator_y,
                                         generator_g_optimizer=self.generator_g_optimizer,
                                         generator_f_optimizer=self.generator_f_optimizer,
                                         discriminator_x_optimizer=self.discriminator_x_optimizer,
                                         discriminator_y_optimizer=self.discriminator_y_optimizer,
                                         step=self.total_steps,
                                         epoch=self.total_epochs)
        return checkpoint

    def restore_latest_checkpoint_if_exists(self):
        """Restores the latest checkpoint from the checkpoint directory."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Using the latest checkpoint of CycleGAN.")
        else:
            print("No checkpoint found for CycleGAN. Starting from scratch.")

    def print_welcome(self):
        """Prints a welcome message to the console."""
        print("""
   ___        _      ___   _   _  _ 
  / __|  _ __| |___ / __| /_\ | \| |
 | (_| || / _| / -_) (_ |/ _ \| .` |
  \___\_, \__|_\___|\___/_/ \_\_|\_|
      |__/                                              
            """)
