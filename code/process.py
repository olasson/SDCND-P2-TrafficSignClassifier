"""
This file contains functions for preparing images for use by a model.
"""

import numpy as np
import cv2

def histogram_equalization(X):

    n_rows, n_cols, n_channels = X[0].shape
    n_images = len(X)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (4,4))

    X_equalized = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = np.uint8)

    for i in range(n_images):
        X_lab = cv2.cvtColor(X[i], cv2.COLOR_RGB2LAB)
        X_lab[:, :, 0] = clahe.apply(X_lab[:, :, 0])
        X_equalized[i] = cv2.cvtColor(X_lab, cv2.COLOR_LAB2RGB)
    
    return X_equalized

def grayscale(X):
      
    n_rows, n_cols, _ = X[0].shape
    n_images = len(X)

    X_grayscale = np.zeros((n_images, n_rows, n_cols), dtype = np.uint8)

    for i in range(n_images):
        X_grayscale[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)

    return X_grayscale


def normalize(X, a, b, X_min, X_max):

        
    X_normalized = a + (((X - X_min) * (b - a)) / (X_max - X_min))

    X_normalized = np.asarray(X_normalized)
    
    return X_normalized

def pre_process(X):

    X = histogram_equalization(X)

    X = grayscale(X)

    X = normalize(X, 0, 1, 0, 255)

    X = X[..., np.newaxis]

    return X