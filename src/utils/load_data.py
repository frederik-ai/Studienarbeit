from PIL import Image
import os

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
