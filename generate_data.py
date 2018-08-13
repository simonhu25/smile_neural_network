"""
Written by Jun Hao Hu. All rights reserved.
"""

import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import glob

NUM_IMAGES = 13718
SPLIT_INDEX = int(np.floor(NUM_IMAGES*0.80))
RANDOM_STATE = 42

def create_data_array(image_path):
    """
    Creates the data array for feeding into the CNN.

    For the training/testing split, we will use an 80/20 training/testing split.

    For the training/validation split, we will use a 80/20 training/validation split.

    Parameters
    ----------

        image_files : str
            A string containing the list of image files.

    Returns
    -------

        data : numpy array
            A numpy array that stores all the image data from the dataset.
    """

    image_files = glob.glob(image_path)
    data = np.array([np.array(cv2.imread(file,0)) for file in image_files])

    return data

def create_label_array(file_path):
    """
    Creates the label array for feeding into the CNN.

    These labels will be encoded using one-hot encoding.

    Parameters
    ----------

        file_path : str
            File location of the .csv file containing the labels.

    Returns
    -------

        labels : Dictionary
    """

    emotions = []

    # Import the labeled emotions and append them to 'emotions'.

    with open(file_path,newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            emotions.append(row['emotion'])

    # Use one-hot encoding to encode these emotions.

    labels = to_categorical(emotions)

    return labels

def main():
    data = create_data_array('./datasets_sandbox/images/*')
    labels = create_label_array('./datasets_sandbox/data/legend.csv')
    print(data.shape)
    print(labels.shape)
