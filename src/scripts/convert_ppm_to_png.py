"""
Script to convert all ppm files in a folder to png files. Is used because training/test data is provided as ppm files.
"""

from PIL import Image
import os


def main():
    IMAGE_PATH = r"C:\Users\Frederik\Documents\Studienarbeit\data\Official Test\Final_Test\Images"

    for file in os.listdir(IMAGE_PATH):
        if file.endswith(".ppm"):
            full_file_path = os.path.join(IMAGE_PATH, file)
            image = Image.open(os.path.join(IMAGE_PATH, file))
            image.save(full_file_path.replace('ppm', 'png'))  # save the ppm file as a png file
            os.remove(os.path.join(IMAGE_PATH, file))


if __name__ == '__main__':
    main()
