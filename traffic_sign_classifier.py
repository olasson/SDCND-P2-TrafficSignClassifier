
import cv2
import numpy as np
import argparse
from os.path import join as path_join

# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, parse_file_path
from code.io import load_config, glob_images
from code.plots import plot_images

FOLDER_DATA = './data'

INFO_PREFIX = 'INFO_MAIN: '
WARNING_PREFIX = 'WARNING_MAIN: '
ERROR_PREFIX = 'ERROR_MAIN: '

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Traffic Sign Classifier')

    # Data

    parser.add_argument(
        '--data_train',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a pickled data file containing training data.',
    )

    parser.add_argument(
        '--data_test',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a pickled data file containing testing data.',
    )

    parser.add_argument(
        '--data_valid',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a pickled data file validation training data.',
    )

    parser.add_argument(
        '--data_meta',
        type = str,
        nargs = '?',
        default = './signnames.csv',
        help = 'File path to a pickled data file containing meta data.',
    )

    parser.add_argument(
        '--show_images',
        action = 'store_true',
        help = 'Shows a set or subset of images of provided data sets.'
    )

    parser.add_argument(
        '--show_distribution',
        action = 'store_true',
        help = 'Shows the label distribution(s) of provided data sets.'
    )

    parser.add_argument(
        '--n_max_cols',
        type = int,
        default = 3,
        help = 'The maximum number of columns in the image plot.'
    )

    # Config

    parser.add_argument(
        '--model_config',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a .json file containing model config.',
    )

    # Misc

    parser.add_argument(
        '--force_save',
        action = 'store_true',
        help = 'If enabled, permits overwriting existing data.'
    )

    args = parser.parse_args()

    # Init paths

    file_path_model_config = args.model_config

    file_path_train = args.data_train
    file_path_test = args.data_test
    file_path_valid = args.data_valid
    file_path_meta = args.data_meta

    # Init config

    model_config = load_config(file_path_model_config)

    # Init values

    n_max_cols = args.n_max_cols

    # Init flags
    
    flag_show_images = args.show_images
    flag_show_distribution = args.show_distribution

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)

    # Show images

    if flag_show_images:

        if folder_is_empty(folder_path_images):
            print(ERROR_PREFIX + 'You are trying to show a set of images but ' + folder_path_images + ' is empty or does not exist!')
            exit()

        print(INFO_PREFIX + 'Showing images from folder: ' + folder_path_images)

        images, file_names = glob_images(folder_path_images)

        plot_images(images, file_names, title_fig_window = folder_path_images, n_max_cols = n_max_cols)

        exit()








