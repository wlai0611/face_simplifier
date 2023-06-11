import numpy as np


def convert_to_greyscale(img):
    #the algorithm to convert a RGB to greyscale described in: https://stackoverflow.com/a/51286918
    color_weights = np.array([0.07, 0.72, 0.21])
    grey_image = np.tensordot(a=img, b=color_weights, axes=(2,0))
    return grey_image