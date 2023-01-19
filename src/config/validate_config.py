import toml
import os

config = toml.load('./config.toml')

# PATHS
assert(isinstance(config['paths']['train_data'], str))
assert(isinstance(config['paths']['pictograms'], str))
assert(os.path.exists(config['paths']['train_data']))
assert(os.path.exists(config['paths']['pictograms']))

# MODEL
image_dimensions = config['model']['image_dimensions']
assert(isinstance(image_dimensions, list))
assert(len(image_dimensions) == 2)
assert(image_dimensions[0] == image_dimensions[1])

# TRAINING
assert(isinstance(config['training']['batch_size'], int))
assert(isinstance(config['training']['number_of_epochs'], int))

print('Successfully validated the configuration file.')