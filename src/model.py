import generator_model
import discriminator_model

class CycleGan:
   
   def __init__(self):
      # GAN 'G'
      self.g_generator = generator_model.make_generator_model()
      self.g_discriminator = discriminator_model.make_discriminator_model()
      # GAN 'F'
      self.f_generator = generator_model.make_generator_model()
      self.f_discriminator = discriminator_model.make_discriminator_model()
   
   def compile(self):
      self.g_generator.compile()
      self.g_discriminator.compile()
      self.f_generator.compile()
      self.f_discriminator.compile()
   
   def propagate(self, x):
      # generated_images = self.g_generator.generator(x)
      # real_or_fake_guesses = self.g_discriminator.discriminator(generated_images)
      # backwards_generated_images = self.f_generator.generator(generated_images)
      # cycle_consistency_guess = self.f_discriminator.discriminator()
      return
   
   def train(self):
      return
