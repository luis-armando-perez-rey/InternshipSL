import subprocess
from os import makedirs, remove
from os.path import join, exists
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from Asbestos_Utils import load_image_label, get_file_extension, load_names

def get_contours(image, invert = True):
    """
    A function that obtains the contours from an image
    :return: A list with the contours for an image
    """
    image = image.astype(np.uint8)
    if invert:
        __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours_per_image