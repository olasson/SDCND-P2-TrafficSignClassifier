
import cv2
import numpy as np
import argparse
import random
from pandas import read_csv
from os.path import join as path_join


# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, pick_random_samples
from code.io import load_config, load_pickled_data
from code.plots import plot_images, plot_distributions

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
        '--distribution_title',
        type = str,
        nargs = '?',
        default = 'Sign distribution.',
        help = 'Title for the distribution plot',
    )

    parser.add_argument(
        '--n_max_images',
        type = int,
        default = 25,
        help = 'The maximum number of images in the image plot.'
    )

    parser.add_argument(
        '--n_max_cols',
        type = int,
        default = 5,
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
    n_max_images = args.n_max_images

    distribution_title = args.distribution_title

    X_train, y_train = None, None
    X_test, y_test = None, None
    X_valid, y_valid = None, None
    y_metadata = None

    # Init flags
    
    flag_show_images = args.show_images
    flag_show_distribution = args.show_distribution

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)

    # Load data

    if file_exists(file_path_train):
        X_train, y_train = load_pickled_data(file_path_train)

    if file_exists(file_path_test):
        X_test, y_test = load_pickled_data(file_path_test)

    if file_exists(file_path_valid):
        X_valid, y_valid = load_pickled_data(file_path_valid)

    if file_exists(file_path_meta):
        y_metadata = read_csv(file_path_meta)['SignName']


    # Show images

    if flag_show_images:

        if (X_train is not None):
            print(INFO_PREFIX + 'Picking random samples from ' + file_path_train + '!')
            X_samples, _, y_meta_samples = pick_random_samples(X_train, y_train, y_metadata, n_max_samples = n_max_images)
            plot_images(X_samples, y_meta_samples, title_fig_window = file_path_train, n_max_cols = n_max_cols)

        if (X_test is not None):
            print(INFO_PREFIX + 'Picking random samples from ' + file_path_test + '!')
            X_samples, _, y_meta_samples = pick_random_samples(X_test, y_test, y_metadata, n_max_samples = n_max_images)
            plot_images(X_samples, y_meta_samples, title_fig_window = file_path_test, n_max_cols = n_max_cols)

        if (X_valid is not None):
            print(INFO_PREFIX + 'Picking random samples from ' + file_path_valid + '!')
            X_samples, _, y_meta_samples = pick_random_samples(X_valid, y_valid, y_metadata, n_max_samples = n_max_images)
            plot_images(X_samples, y_meta_samples, title_fig_window = file_path_valid, n_max_cols = n_max_cols)

    # Show distributions

    if flag_show_distribution:
        print(INFO_PREFIX + 'Showing sign label distribution(s)!')
        plot_distributions(y_train, y_test, y_valid, y_metadata, title = distribution_title)










