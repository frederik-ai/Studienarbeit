from PIL import Image
import os
import tensorflow as tf

def convert_ppm_training_data_to_png(path_to_train_directory):
    # get full system path of directory where training data is located
    train_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', path_to_train_directory))

    # iterate through each level-1 subdirectory
    for folder in os.listdir(train_directory):
        # iterate through each ppm file in the current subdirectory
        for file in os.listdir(train_directory + '\\' + folder):
            filename = os.fsdecode(file)
            full_filepath = train_directory + '\\' + folder + '\\' + filename
            if filename.endswith('.ppm'):
                img = Image.open(full_filepath)
                img.save(full_filepath.replace('ppm', 'png'))   # save the ppm file as a png file
                os.remove(full_filepath)                        # delete the ppm file

def load_training_data():
    return

def load_test_data(path_to_test_directory, batch_size=32, image_dimensions=(128,128)):
    test_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', path_to_test_directory))
    x_test = tf.keras.utils.image_dataset_from_directory(test_path, batch_size=batch_size, image_size=image_dimensions,
                                                      labels=None)
    return x_test 
    
def normalize_dataset(dataset):
    return dataset.map(lambda tensor: (tensor/127.5)-1)