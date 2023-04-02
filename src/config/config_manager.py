import toml
import os

_config = toml.load('./config/config.toml')


def validate_config():
    """
        Validate the configuration file.

        Returns:
            bool: True if the configuration file is valid, otherwise function throws an exception.
        """
    # config = toml.load('./config.toml')

    # PATHS
    path_variables = ['train_data', 'pictograms', 'augmentation_data', 'destination']
    for path_variable in path_variables:
        assert (isinstance(_config['paths'][path_variable], str))
        # TODO: check if path exists

    # MODEL
    image_size = _config['model']['image_size']
    assert (isinstance(image_size, int))

    # TRAINING
    assert (isinstance(_config['training']['batch_size'], int))
    assert (isinstance(_config['training']['number_of_epochs'], int))

    return 1


def get_config():
    """
        Expose the configuration file. Configuration file is validated before it is returned.
        """
    if validate_config():
        return _config
    else:
        raise ValueError('Configuration file contains invalid values. See above exception(s) for details.')
