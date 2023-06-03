"""
Script for validating the configuration file.
"""

import toml
import os


def main():
    config = toml.load('./config.toml')

    # PATHS
    paths = ['train_data', 'pictograms', 'augmentation_data', 'destination']
    classifier_paths = ['train_data', 'test_data']
    for path in paths:
        assert (path in config['paths'].keys())
        assert (isinstance(config['paths'][path], str))
        # assert (os.path.exists(os.path.join('..', config['paths'][path])))
    for path in classifier_paths:
        assert (path in config['paths']['classifier'].keys())
        assert (isinstance(config['paths']['classifier'][path], str))
        # assert (os.path.exists(config['paths']['classifier'][path]))
    assert (isinstance(config['paths']['classifier']['checkpoint_directory_name'], str))

    # MODEL
    assert (isinstance(config['model']['image_size'], int))
    assert (isinstance(config['model']['generator_type'], str))

    # TRAINING
    integers = ['number_of_epochs', 'batch_size', 'lambda']
    floats = ['learning_rate', 'beta_1', 'beta_2']
    for integer in integers:
        assert (isinstance(config['training'][integer], int))
    for float_ in floats:
        assert (isinstance(config['training'][float_], float))

    print('Successfully validated the configuration file.')


if __name__ == '__main__':
    main()
