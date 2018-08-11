'''
Written by Jun Hao Hu. All rights reserved.
'''

import cv2
import glob
import csv
import numpy as np
from scipy import ndimage

def preprocess_images(folder_location):
    """
    Summary: Preprocess images for feeding into a CNN.

    Parameters
    ----------

    folder_location : str
        Location of the folder containing the images to be proprocessed.

    Returns
    -------

    Returns 0 if process is successful, 1 otherwise.
    """

    image_files = glob.glob(folder_location)

    """ Resize the images to 48 x 48 pixels, normalize them, and write those images into the folder. """

    index = 1
    for image_path in image_files:
        image = cv2.imread(image_path,0)
        image = cv2.resize(image,(48,48),interpolation=cv2.INTER_LANCZOS4)
        image = image/255
        image_blurred = ndimage.gaussian_filter(image,3)
        cv2.imwrite(image_path,image_blurred)
        print("Image no.",index)
        index += 1

def main():
    preprocess_images("./datasets_sandbox/images/*")

if __name__ == "__main__":
    main()
