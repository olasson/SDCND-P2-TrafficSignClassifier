"""
This file contains some miscellaneous helper functions and os wrappers.
"""

import os
import numpy as np

def file_exists(file_path):
    """
    Check if a file exists.
    
    Inputs
    ----------
    path: str
        Path to file.
       
    Outputs
    -------
    bool
        True if file exists, false otherwise.
        
    """

    if file_path is None:
        return False

    if not os.path.isfile(file_path):
        return False

    return True

def folder_guard(folder_path):
    """
    Checks if a folder exists and creates it if it does not.
    
    Inputs
    ----------
    folder_path: str
        Path to folder.
       
    Outputs
    -------
        N/A
        
    """
    if not os.path.isdir(folder_path):
        print('INFO:folder_guard(): Creating folder: ' + folder_path + '...')
        os.mkdir(folder_path)

def folder_is_empty(folder_path):
    """
    Check if a folder is empty. If the folder does not exist, it counts as being empty. 
    
    Inputs
    ----------
    folder_path: str
        Path to folder.
       
    Outputs
    -------
    bool
        True if folder exists and contains elements, false otherwise.
        
    """

    if os.path.isdir(folder_path):
        return (len(os.listdir(folder_path)) == 0)
    
    return True

def parse_file_path(file_path):

    """
    Parse out the folder path and file path from a full path.
    
    Inputs
    ----------
    file_path: string
        Path to a file - './path/to/myfile.jpg'
        
    Outputs
    -------
    folder_path: string
        The folder path contained in 'file_path' - './path/to/'
    file_name: string
        The file_name contained in 'file_path' - 'myfile.jpg'
    """

    file_name = os.path.basename(file_path)

    cutoff = len(file_path) - len(file_name)

    folder_path = file_path[:cutoff]

    return folder_path, file_name

# Samples

def pick_samples_X(X, indices):

    n_samples = len(indices)

    n_rows, n_cols, n_channels = X[0].shape

    X_samples = np.zeros((n_samples,n_rows, n_cols, n_channels), dtype = np.uint8)

    for i, index in enumerate(indices):
        X_samples[i] = X[index]

    return X_samples

def pick_samples_y(y, indices, y_metadata = None):

    n_samples = len(indices)

    y_samples = np.zeros((n_samples), dtype = np.int)

    for i, index in enumerate(indices):
        y_samples[i] = y[index]

    y_metadata_samples = None

    if y_metadata is not None:
        y_metadata_samples = np.zeros((n_samples), dtype = 'U25') # String
        for i in range(n_samples):
            y_metadata_samples[i] = y_metadata[y_samples[i]]

    return y_samples, y_metadata_samples

def pick_random_samples(X, y, y_metadata, n_max_samples = 25):

    n_samples = len(X)

    indices = np.random.randint(0, n_samples, min(n_samples, n_max_samples))

    X_samples = pick_samples_X(X, indices)
    y_samples, y_metadata_samples = pick_samples_y(y, indices, y_metadata)

    return X_samples, y_samples, y_metadata_samples

def distribution_is_uniform(y): 

    if y is None:
        return True

    is_uniform = True
    
    classes, classes_count = np.unique(y, return_counts = True)

    class_ref_count = classes_count[0]

    for class_count in classes_count:

        if class_count != class_ref_count:
            is_uniform = False
            break

    return is_uniform



        







