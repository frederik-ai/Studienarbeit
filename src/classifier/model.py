import tensorflow as tf


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


def main():
    batch_size = 32
    train, val = tf.keras.utils.image_dataset_from_directory(r'C:\Users\Frederik\Desktop\data\Test', batch_size=batch_size,
                                                          validation_split=0.2, subset='both', seed=123,
                                                          image_size=(256, 256), label_mode='categorical')

    model = create_model()
    model.fit(x=train, validation_data=val, epochs=5)


if __name__ == '__main__':
    main()
