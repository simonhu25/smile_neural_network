"""
Written by Jun Hao Hu. All rights reserved. 
"""

import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import sys

def sharpen_image(alpha):
    """
    Sharpens the image according to the alpha value.
    """

    image = cv2.imread('./datasets_sandbox/images/Aaron_Eckhart_0001.jpg',0)
    image_blurred = ndimage.gaussian_filter(image,3)
    image_blurred_filter = ndimage.gaussian_filter(image_blurred,1)
    image_sharpened = image_blurred + alpha * (image_blurred - image_blurred_filter)

    plt.figure()

    plt.subplot(131)
    plt.imshow(image,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(image_blurred,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(image_sharpened,cmap=plt.cm.gray)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    sharpen_image(int(sys.argv[1]))

if __name__ == "__main__":
    main()
