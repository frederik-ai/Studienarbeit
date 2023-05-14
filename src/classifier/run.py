"""
Train or evaluate a traffic sign classifier.

Usage:
    `$ python run.py (--train | --evaluate)`

Options:

Example:
    `$ python run.py --train`
    `$ python run.py --evaluate`
"""
import argparse
import os
import uuid
import toml

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

tf.get_logger().setLevel('ERROR')

config = toml.load('../config/config.toml')
num_epochs = 20
batch_size = 32


def run():
    parser = argparse.ArgumentParser()
    # Determine whether to train or evaluate the model
    train_or_eval = parser.add_mutually_exclusive_group(required=True)
    train_or_eval.add_argument("--train", default=False, action='store_true',
                               help="Train the classifier")
    train_or_eval.add_argument("--evaluate", default=False, action='store_true',
                               help="Evaluate the trained classifier")
    args = parser.parse_args()

    checkpoint_directory = f"checkpoints/{config['paths']['classifier']['checkpoint_directory_name']}"
    checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_directory)

    model = get_model()

    # Load the latest checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint is not None:
        model.load_weights(latest_checkpoint)
        print(f'Loaded checkpoint: {latest_checkpoint}')
    else:
        print('No checkpoint found, hence no weights were loaded.')

    val_set = tf.keras.utils.image_dataset_from_directory(config['paths']['classifier']['test_data'],
                                                          batch_size=batch_size, seed=123,
                                                          image_size=(256, 256), label_mode='categorical',
                                                          crop_to_aspect_ratio=True)

    if args.train:
        print('Training...')
        train_set = tf.keras.utils.image_dataset_from_directory(config['paths']['classifier']['train_data'],
                                                                batch_size=batch_size, seed=123,
                                                                image_size=(256, 256), label_mode='categorical',
                                                                crop_to_aspect_ratio=True)
        model.fit(x=train_set, validation_data=val_set, epochs=20)
        model.save_weights(checkpoint_path)
    elif args.evaluate:
        print('Evaluating...')
        model.evaluate(val_set)
    else:
        raise ValueError('No argument for train or evaluate was given.')


def get_model():
    num_classes = 43
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create a new model with the modified architecture
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with a low learning rate
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train(checkpoint_path, model, train_set, val_set):
    print('Training...')

    train_set = tf.keras.utils.image_dataset_from_directory(config['paths']['classifier']['train_data'],
                                                            batch_size=batch_size, seed=123,
                                                            image_size=(256, 256), label_mode='categorical')

    val_set = tf.keras.utils.image_dataset_from_directory(config['paths']['classifier']['test_data'],
                                                             batch_size=batch_size, seed=123,
                                                             image_size=(256, 256), label_mode='categorical')

    model.fit(x=train_set, validation_data=val_set, epochs=num_epochs)
    model.save_weights(checkpoint_path)


def main():
    run()


if __name__ == '__main__':
    main()


def create_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
