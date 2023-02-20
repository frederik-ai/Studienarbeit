"""
Remove training/test images that have a height or width smaller than a certain threshold.
"""

import cv2
import pathlib
import os


def main():
    absolute_imgs_path = ''
    max_img_height = 50
    max_img_width = 50

    for path, subdirs, file_names in os.walk(absolute_imgs_path):
        for file in file_names:
            if file.lower().endswith('.png'):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                dimensions = img.shape
                height = img.shape[0]
                width = img.shape[1]
                if (width < max_img_width) or (height < max_img_height):
                    os.remove(img_path)


if __name__ == '__main__':
    main()
