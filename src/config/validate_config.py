import toml

config = toml.load('./config.toml')

# PATHS
assert(isinstance(config['paths']['train_data'], str))
assert(isinstance(config['paths']['pictograms'], str))

# MODEL
assert(isinstance(config['model']['image_dimensions'], list))

# TRAINING
assert(isinstance(config['training']['batch_size'], int))
assert(isinstance(config['training']['number_of_epochs'], int))

print('Successfully validated the configuration file.')