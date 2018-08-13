"""
Written by Jun Hao Hu. All rights reserved.
"""

import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import glob

NUM_IMAGES = 13718

def create_data_array(image_path):
    """
    Creates the data array for feeding into the CNN.

    For the training/testing split, we will use an 80/20 training/testing split.

    For the training/validation split, we will use a 80/20 training/validation split.t

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

    # Note to self: The above importing works. Make sure you turn everything to lower case before you import it; otherwise there
    # will be issues with the readability of the emotion.
    # However, the below to_categorical does not work. So either use another module or write
    # your own code for this one.

    labels = to_categorical(emotions)

    return labels

def split_data(data, labels, split_percent_training, split_percent_validation, random_state):
    """
    Split the data into training, validation, and test data sets.

    Parameters
    ----------

        data : numpy array
            Numpy array that contains the unsplit data.

        labels : numpy array
            Numpy array that contains the unsplit labels.

        split_percent_training : int
            Percent of the data that should be left to training.

        split_percent_validation : int
            Percent of the training data that should be left to validation.

        random_state : numpy array
            Permutation matrix that permutes the data.

    Returns
    -------

        x_train : numpy array
            Numpy array containing the training images.

        x_val : numpy array
            Numpy array containing the validation images.

        x_test : numpy array
            Numpy array that contains the test images.

        y_train : numpy array
            Numpy array that contains the training labels.

        y_val : numpy array
            Numpy array that contains the validation labels.

        y_test : numpy array
            Numpy array that contains the test labels.

    """



def main():
    data = create_data_array('./datasets_sandbox/images/*')
    labels = create_label_array('./datasets_sandbox/data/legend.csv')
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data, labels, 0.80, 0.80, 42)
    print(data.shape)
    print(labels.shape)
