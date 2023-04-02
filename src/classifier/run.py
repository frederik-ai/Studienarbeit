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

config = toml.load('../config/config.toml')
num_epochs = 10


def run():
    parser = argparse.ArgumentParser()
    # Determine whether to train or evaluate the model
    train_or_eval = parser.add_mutually_exclusive_group(required=True)
    train_or_eval.add_argument("--train", default=False, action='store_true',
                               help="Train the classifier")
    train_or_eval.add_argument("--evaluate", default=False, action='store_true',
                               help="Evaluate the trained classifier")
    args = parser.parse_args()

    checkpoint_path = (os.path.join(os.path.dirname(__file__), 'checkpoints'))

    if args.train:
        train(checkpoint_path)
    elif args.evaluate:
        evaluate(checkpoint_path)
    else:
        raise ValueError('No argument for train or evaluate was given.')


def get_model():
    num_classes = 43
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

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


def train(checkpoint_path):
    batch_size = 32

    # vassert (use_real_data or use_unet_data or use_resnet_data)
    # train_set = None
    # if use_real_data:
    #     if train_set is None:
    #         train_set = get_image_dataset(r'C:\Users\Frederik\Desktop\data\Train')
    #     else:
    #         train_set.concatenate(get_image_dataset(r'C:\Users\Frederik\Desktop\data\Train'))

    train_set, val_set = tf.keras.utils.image_dataset_from_directory(r'C:\Users\Frederik\Desktop\data\Test',
                                                             batch_size=batch_size,
                                                             validation_split=0.2, subset='both', seed=123,
                                                             image_size=(256, 256), label_mode='categorical')

    model = create_model()
    model.fit(x=train_set, validation_data=val_set, epochs=5)
    print('Training...')


def evaluate(checkpoint_path):
    print('Evaluating...')


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
