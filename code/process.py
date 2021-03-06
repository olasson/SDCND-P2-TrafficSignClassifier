"""
This file contains functions for preparing images for use by a model.
"""

import numpy as np
import cv2

def histogram_equalization(images):
    """
    Apply histogram equalization to a set of images.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images.
        
    Outputs
    -------
    image_equalized: numpy.ndarray
        Numpy array containing histogram equalized images.
    """

    n_rows, n_cols, n_channels = images[0].shape
    n_images = len(images)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (4,4))

    image_equalized = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = np.uint8)

    for i in range(n_images):
        image_lab = cv2.cvtColor(images[i], cv2.COLOR_RGB2LAB)
        image_lab[:, :, 0] = clahe.apply(image_lab[:, :, 0])
        image_equalized[i] = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    
    return image_equalized

def grayscale(images):
    """
    Apply the grayscale transformation to a set of images.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images.
        
    Outputs
    -------
    image_grayscale: numpy.ndarray
        Numpy array containing grayscale images.
    """

    n_rows, n_cols, _ = images[0].shape
    n_images = len(images)

    image_grayscale = np.zeros((n_images, n_rows, n_cols), dtype = np.uint8)

    for i in range(n_images):
        image_grayscale[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    return image_grayscale


def normalize(images, a, b, image_min, image_max):
    """
    Apply min/max feature scaling to a set of images.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images.
    a,b: int, int
        The desired numerical range for the scaled images.
    image_min, image_max: int,int
        The min/max numberical value in 'images'
        
    Outputs
    -------
    image_grayscale: numpy.ndarray
        Numpy array containing images scaled to [a,b].
    """
        
    images_normalized = a + (((images - image_min) * (b - a)) / (image_max - image_min))

    images_normalized = np.asarray(images_normalized)
    
    return images_normalized

def pre_process(images):
    """
    Apply pre-processing to a set of images.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images.
        
    Outputs
    -------
    image_grayscale: numpy.ndarray
        Numpy array containing images ready for use by a model.
    """

    images = histogram_equalization(images)

    images = grayscale(images)

    images = normalize(images, 0, 1, 0, 255)

    images = images[..., np.newaxis]

    return images