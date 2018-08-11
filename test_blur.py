"""
Written by Jun Hao Hu. All rights reserved.
"""

import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def blur_image(image_path):
    """ Blur the image using Gaussian filtering. """

    image = cv2.imread(image_path,0)
    image_blurred_3 = ndimage.gaussian_filter(image,3)
    image_blurred_5 = ndimage.gaussian_filter(image,5)
    image_blurred_7 = ndimage.gaussian_filter(image,7)
    image_blurred_9 = ndimage.gaussian_filter(image,9)

    plt.figure()

    plt.subplot(151)
    plt.imshow(image,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(image_blurred_3,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(153)
    plt.imshow(image_blurred_5,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(154)
    plt.imshow(image_blurred_7,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(155)
    plt.imshow(image_blurred_9,cmap=plt.cm.gray)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    blur_image('./datasets_sandbox/images/Aaron_Pena_0001.jpg')

if __name__ == "__main__":
    main()
