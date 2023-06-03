"""
Run traffic sign image generation with a trained CycleGAN model.

Usage:
    `$ python generate.py [--num-imgs <int>] [--o <str>] [--model <str>] [--make-invalid] [--motion-blur] [--snow]`

Options:
    `[--num-imgs <int>]`       Number of generated images. <br>
    `[--o <str>]`           Path to directory where generated images are stored. Default: value from config file. <br>
    `[--model <str>]`       Model name [unet or resnet]. Default: value from config file. <br>
    `[--make-invalid]`      Make generated street signs invalid. <br>
    `[--motion-blur]`       Add random motion blur to the generated images. <br>
    '[--snow]'              Add random snow to the generated images. <br>

Example:
    `$ python generate.py --model unet --num-imgs 10 --make-invalid`
"""


import uuid
import argparse
import math
import os
import tensorflow as tf
from tqdm import tqdm
import toml

import utils.load_data
import utils.preprocess_image
import utils.misc
import utils.image_augmentation
import model


def main():
    # Setup
    config = toml.load('./config/config.toml')
    batch_size = config['training']['batch_size']
    image_size = config['model']['image_size']

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-imgs', type=int, default=5, help='number of generated images')
    parser.add_argument('--o', type=str, default=config['paths']['destination'],
                        help='path to directory where generated images are stored')
    parser.add_argument('--model', type=str, default=config['model']['generator_type'],
                        help='model name [unet or resnet]')
    parser.add_argument('--make-invalid', dest='make_invalid', default=False, action='store_true',
                        help='make street signs invalid')
    parser.add_argument('--motion-blur', dest='motion_blur', default=False, action='store_true',
                        help='add random motion blur')
    parser.add_argument('--snow', dest='snow', default=False, action='store_true', help='add snow')
    args = parser.parse_args()
    if args.model == 'unet' or args.model == 'resnet':
        config['model']['generator_type'] = args.model
    else:
        raise ValueError('model must be "unet" or "resnet". This argument is optional')
    destination_path = args.o
    num_generated_images = args.num_imgs
    snow = args.snow
    make_signs_invalid = args.make_invalid
    motion_blur = args.motion_blur

    path_to_pictograms = config['paths']['pictograms']
    x_pictograms = tf.keras.utils.image_dataset_from_directory(path_to_pictograms, batch_size=batch_size,
                                                               image_size=(image_size, image_size), labels=None)
    x_pictograms_processed = utils.load_data.normalize_dataset(x_pictograms)

    # Currently, batch size cannot be larger than the number of pictograms
    number_of_pictograms = x_pictograms_processed.cardinality().numpy()  # number of elements in pictogram dataset
    batch_size = min(batch_size, number_of_pictograms)

    cycle_gan = model.CycleGan(config)
    cycle_gan.compile()
    cycle_gan.restore_latest_checkpoint_if_exists()

    # Generate images
    print('Generating images...')
    num_iterations = math.ceil(num_generated_images / batch_size)
    num_generated_images_left = num_generated_images
    for iteration in tqdm(range(num_iterations)):
        x_pictograms_processed.shuffle(buffer_size=100, reshuffle_each_iteration=True)

        # Transform pictograms
        single_pictogram_batch = x_pictograms_processed.take(1).get_single_element()
        single_pictogram_batch, content_sizes, transform_matrices = utils.preprocess_image.randomly_transform_image_batch(
            single_pictogram_batch)
        transform_matrices = [matrix for matrix in transform_matrices]  # convert to list to be able to pop() it

        # Put transformed pictograms into the generator
        generator_result = cycle_gan.generate(single_pictogram_batch)

        if make_signs_invalid:
            cross_img_path = config['paths']['augmentation_data'] + '/cross.png'
            generator_result = tf.map_fn(
                lambda t: utils.image_augmentation.make_street_sign_invalid(t, cross_img_path, content_sizes.pop(0),
                                                                            transform_matrices.pop(0)),
                generator_result)
        if snow:
            generator_result = tf.map_fn(lambda t: utils.image_augmentation.add_snow(t, 20, 30), generator_result)
            generator_result = tf.map_fn(lambda t: utils.image_augmentation.add_snow(t, 10, 20), generator_result)
        if motion_blur:
            generator_result = tf.map_fn(lambda t: utils.image_augmentation.apply_random_motion_blur(t),
                                         generator_result)
        if batch_size < num_generated_images_left:
            num_images = batch_size
            num_generated_images_left -= batch_size
        else:
            num_images = num_generated_images_left
        for i in range(num_images):
            utils.misc.store_tensor_as_img(generator_result[i, :], str(uuid.uuid4()), destination_path)


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
