from PIL import Image
import os

IMAGE_PATH = r"C:\Users\Frederik\Documents\Studienarbeit\data\Official Test\Final_Test\Images"

for file in os.listdir(IMAGE_PATH):
    if file.endswith(".ppm"):
        full_file_path = os.path.join(IMAGE_PATH, file)
        image = Image.open(os.path.join(IMAGE_PATH, file))
        image.save(full_file_path.replace('ppm', 'png'))  # save the ppm file as a png file
        os.remove(os.path.join(IMAGE_PATH, file))
