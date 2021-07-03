
import cv2
import numpy as np
import argparse
import random
from pandas import read_csv
from os.path import join as path_join


# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, pick_random_samples
from code.io import load_config, load_pickled_data, save_pickled_data, load_labels
from code.plots import plot_images, plot_distributions
from code.process import pre_process

FOLDER_DATA = './data'
FOLDER_MODELS = './models'

FILE_PATH_RAW_TRAIN = './data/train.p'
FILE_PATH_RAW_TEST = './data/test.p'
FILE_PATH_RAW_VALID = './data/valid.p'

INFO_PREFIX = 'INFO_MAIN: '
WARNING_PREFIX = 'WARNING_MAIN: '
ERROR_PREFIX = 'ERROR_MAIN: '

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Traffic Sign Classifier')

    # Data

    parser.add_argument(
        '--data_meta',
        type = str,
        nargs = '?',
        default = './signnames.csv',
        help = 'File path to a pickled data file containing meta data.',
    )

    parser.add_argument(
        '--show_images',
        type = str,
        nargs = '?',
        default = None,
        help = 'File path to a pickled (.p) file containing a set of images.',
    )

    parser.add_argument(
        '--show_distributions',
        type = str,
        nargs = '+',
        default = None,
        help = 'File path(s) to pickled file(s) containing a label set.',
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

    file_path_meta = args.data_meta

    file_path_images = args.show_images
    file_path_distributions = args.show_distributions #if (args.show_distributions is None) else args.show_distributions

    # Init config

    model_config = load_config(file_path_model_config)

    # Init values

    n_max_cols = args.n_max_cols
    n_max_images = args.n_max_images

    distribution_title = args.distribution_title

    #X_train, y_train = None, None
    #X_test, y_test = None, None
    #X_valid, y_valid = None, None
    y_metadata = None


    # Init flags
    

    flag_force_save = args.force_save

    # Folder setup

    folder_guard(FOLDER_DATA)
    folder_guard(FOLDER_MODELS)

    # Load data

    if file_exists(file_path_meta):
        y_metadata = read_csv(file_path_meta)['SignName']
    
    # Show images

    if file_path_images is not None:
        print(INFO_PREFIX + 'Showing images from :' + file_path_images)
        X, y = load_pickled_data(file_path_images)
        X_samples, _, y_meta_samples = pick_random_samples(X, y, y_metadata, n_max_samples = n_max_images)
        plot_images(X_samples, y_meta_samples, title_fig_window = file_path_images, n_max_cols = n_max_cols)

        exit()

    # Show distributions

    if file_path_distributions is not None:
        print(INFO_PREFIX + 'Showing sign label distribution(s)!')
        y1, y2, y3 = load_labels(file_path_distributions)
        plot_distributions(y1, y2, y3, y_metadata, title = distribution_title)

    if model_config is not None:

        print(INFO_PREFIX + 'Using config file: ' + file_path_model_config)

        # Paths

        file_path_train_prepared = model_config["data_train"]
        file_path_test_prepared = model_config["data_test"]
        file_path_valid_prepared = model_config["data_valid"]

        file_path_model = model_config["model_name"]

        # Flags

        flag_mirror_data = model_config["mirror_data"]
        flag_transform_data = model_config["transform_data"]

        if file_exists(file_path_train_prepared):
            pass


        # Train

        if not file_exists(file_path_train_prepared):

            X_train, y_train = load_pickled_data(FILE_PATH_RAW_TRAIN)

            if flag_mirror_data:
                print(INFO_PREFIX + 'Mirroring training data!')

            if flag_transform_data:
                print(INFO_PREFIX + 'Applying random transforms to training data!')

            print(INFO_PREFIX + 'Pre-processing training data!')
            X_train = pre_process(X_train)
            save_pickled_data(file_path_train_prepared, X_train, y_train)

        # Test

        if not file_exists(file_path_test_prepared):
            print(INFO_PREFIX + 'Pre-processing test data!')
            X_test, y_test = load_pickled_data(FILE_PATH_RAW_TEST)
            X_test = pre_process(X_test)
            save_pickled_data(file_path_test_prepared, X_test, y_test)
        
        # Valid

        if not file_exists(file_path_valid_prepared):
            print(INFO_PREFIX + 'Pre-processing validation data!')
            X_valid, y_valid = load_pickled_data(FILE_PATH_RAW_VALID)
            X_valid = pre_process(X_valid)
            save_pickled_data(file_path_valid_prepared, X_valid, y_valid)



        








