"""
Written by Jun Hao Hu. All rights reserved.
"""

import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import glob

num_images = 13718

def create_data_array(image_files):
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

        x_train : numpy array
            Numpy array containing all the training images.

        x_val : numpy array
            Numpy array containing all the validation images.

        x_test : numpy array
            Numpy array containing all the testing images.
    """

    image_files = glob.glob('./datasets_sandbox/images/*')
    data = np.array([np.array(cv2.imread(file,0)) for file in image_files])

    index = int(np.floor(num_images*0.80))

    x_train = data[0:index]
    x_test = data[index:num_images+1]

    x_val = x_train[0:index]
    x_train = x_train[index:x_train.shape[0]+1]

    return x_train,x_val,x_test

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

        label_train : numpy array
            Numpy array containing the training labels.

        label_val : numpy array
            Numpy array containing the validation labels.

        label_test : numpy array
            Numpy array containing the testing labels.
    """

    
