"""
Run traffic sign image generation with a trained CycleGAN model.

Usage:
    `$ python generate.py --num-imgs <int> [--o <str>] [--model <str>] [--make-invalid] [--motion-blur]`

Options:
    `--num-imgs <int>`      Number of generated images. <br>
    `[--o <str>]`             Path to directory where generated images are stored. Default: value from config file. <br>
    `[--model <str>]`         Model name [unet or resnet]. Default: value from config file. <br>
    `[--make-invalid]`        Make generated street signs invalid. <br>
    `[--motion-blur]`         Add random motion blur to the generated images. <br>

Example:
    `$ python generate.py --model unet --num-imgs 10 --make-invalid`
"""

import tensorflow as tf
import uuid
import argparse
import math
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
    BATCH_SIZE = config['training']['batch_size']
    IMAGE_SIZE = config['model']['image_size']

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
    args = parser.parse_args()
    if args.model == 'unet' or args.model == 'resnet':
        config['model']['generator_type'] = args.model
    else:
        raise ValueError('model must be "unet" or "resnet". This argument is optional')
    DESTINATION_PATH = args.o
    NUM_GENERATED_IMAGES = args.num_imgs
    MAKE_SIGNS_INVALID = args.make_invalid
    MOTION_BLUR = args.motion_blur

    # PATH_TO_PICTOGRAMS = r'C:\Users\Frederik\Documents\Studienarbeit\data\Pictograms'
    PATH_TO_PICTOGRAMS = config['paths']['pictograms']
    x_pictograms = tf.keras.utils.image_dataset_from_directory(PATH_TO_PICTOGRAMS, batch_size=BATCH_SIZE,
                                                               image_size=(IMAGE_SIZE, IMAGE_SIZE), labels=None)
    x_pictograms_processed = utils.load_data.normalize_dataset(x_pictograms)

    cycle_gan = model.CycleGan(config)
    cycle_gan.compile()
    cycle_gan.restore_latest_checkpoint_if_exists()

    # Generate images
    print('Generating images...')
    num_iterations = math.ceil(NUM_GENERATED_IMAGES / BATCH_SIZE)
    num_generated_images_left = NUM_GENERATED_IMAGES
    for iteration in tqdm(range(num_iterations)):
        x_pictograms_processed.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        single_pictogram_batch = x_pictograms_processed.take(1).get_single_element()
        single_pictogram_batch, content_sizes, transform_matrices = utils.preprocess_image.randomly_transform_image_batch(
            single_pictogram_batch)
        transform_matrices = [matrix for matrix in transform_matrices]  # convert to list to be able to pop() it
        generator_result = cycle_gan.generator_g(single_pictogram_batch, training=False)

        if MAKE_SIGNS_INVALID:
            cross_img_path = config['paths']['augmentation_data'] + '/cross.png'
            generator_result = tf.map_fn(
                lambda t: utils.image_augmentation.make_street_sign_invalid(t, cross_img_path, content_sizes.pop(0),
                                                                            transform_matrices.pop(0)),
                generator_result)
        if MOTION_BLUR:
            generator_result = tf.map_fn(lambda t: utils.image_augmentation.apply_motion_blur(t, 100), generator_result)

        if BATCH_SIZE < num_generated_images_left:
            num_images = BATCH_SIZE
            num_generated_images_left -= BATCH_SIZE
        else:
            num_images = num_generated_images_left
        for i in range(num_images):
            utils.misc.store_tensor_as_img(generator_result[i, :], str(uuid.uuid4()), DESTINATION_PATH)


if __name__ == '__main__':
    main()
