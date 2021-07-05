
import cv2
import numpy as np
import argparse
import random
from pandas import read_csv
from os.path import join as path_join


# Custom imports
from code.misc import file_exists, folder_guard, folder_is_empty, pick_random_samples, pick_samples_images, parse_file_path
from code.io import load_config, load_pickled_data, save_pickled_data, load_labels
from code.plots import plot_images, plot_distributions, plot_model_history, plot_predicitons
from code.process import pre_process
from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.models import train_model, save_model, load_model, evaluate_model, predict_signs

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
    file_path_distributions = args.show_distributions

    # Init config

    model_config = load_config(file_path_model_config)

    # Init values

    n_max_cols = args.n_max_cols
    n_max_images = args.n_max_images

    distribution_title = args.distribution_title

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
        X, y, X_samples, y_samples = None, None, None, None

    # Show distributions

    if file_path_distributions is not None:
        print(INFO_PREFIX + 'Showing sign label distribution(s)!')
        y1, y2, y3 = load_labels(file_path_distributions)
        plot_distributions([y1, y2, y3], y_metadata, title = distribution_title, title_fig_window = 'label_distributions')
        y1, y2, y3 = None, None, None

    if model_config is not None:

        print(INFO_PREFIX + 'Using config file: ' + file_path_model_config)

        # Paths

        file_path_train_prepared = model_config["data_train"]
        file_path_test_prepared = model_config["data_test"]
        file_path_valid_prepared = model_config["data_valid"]

        # Flags

        flag_mirror_data = model_config["mirror_data"]
        flag_transform_data = model_config["transform_data"]



        # Training data

        if not file_exists(file_path_train_prepared):

            X_train, y_train = load_pickled_data(FILE_PATH_RAW_TRAIN)

            if flag_mirror_data:
                print(INFO_PREFIX + 'Mirroring training data!')

                MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                              -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
                              19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
                              30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
                              -1, -1, -1]
                              
                X_train, y_train = augment_data_by_mirroring(X_train, y_train, MIRROR_MAP)

            if flag_transform_data:
                print(INFO_PREFIX + 'Applying random transforms to training data!')
                X_train, y_train = augment_data_by_random_transform(X_train, y_train)

            print(INFO_PREFIX + 'Pre-processing training data!')
            X_train = pre_process(X_train)
            save_pickled_data(file_path_train_prepared, X_train, y_train)
        else:
            print(INFO_PREFIX + 'Loading prepared training data!')
            X_train, y_train = load_pickled_data(file_path_train_prepared)

        # Testing data

        if not file_exists(file_path_test_prepared):
            print(INFO_PREFIX + 'Pre-processing test data!')
            X_test, y_test = load_pickled_data(FILE_PATH_RAW_TEST)
            X_test = pre_process(X_test)
            save_pickled_data(file_path_test_prepared, X_test, y_test)
        else:
            print(INFO_PREFIX + 'Loading prepared testing data!')
            X_test, y_test = load_pickled_data(file_path_test_prepared)
        
        # Validation data

        if not file_exists(file_path_valid_prepared):
            print(INFO_PREFIX + 'Pre-processing validation data!')
            X_valid, y_valid = load_pickled_data(FILE_PATH_RAW_VALID)
            X_valid = pre_process(X_valid)
            save_pickled_data(file_path_valid_prepared, X_valid, y_valid)
        else:
            print(INFO_PREFIX + 'Loading prepared validation data!')
            X_valid, y_valid = load_pickled_data(file_path_valid_prepared)

        # Model

        file_path_model = model_config["model_name"]
        batch_size = model_config["batch_size"]

        # Get model name without extension
        model_name = parse_file_path(file_path_model)[1]
        model_name = model_name[:len(model_name) - len('.h5')]

        if not file_exists(file_path_model):
            
            lrn_rate = model_config["lrn_rate"]
            n_max_epochs = model_config["n_max_epochs"]

            model, history = train_model(file_path_model, X_train, y_train, X_valid, y_valid, lrn_rate, n_max_epochs, batch_size)

            if model is None:
                print(ERROR_PREFIX + 'Unknown model type! The model name: ' + model_name + ' must contain the substring LeNet or VGG16!')
                exit()
            save_model(file_path_model, model)

            plot_model_history(model_name, history, FOLDER_MODELS, lrn_rate, batch_size, n_max_epochs)

        else:
            print(INFO_PREFIX + 'Loading model: ' + file_path_model)
            model = load_model(file_path_model)

        #evaluate_model(model, X_test, y_test, batch_size)

        # Predict on test images

        X_test_raw, _ = load_pickled_data(FILE_PATH_RAW_TEST)

        n_images = len(X_test)

        indices = np.random.randint(0, n_images, min(n_images, n_max_images))

        signs = predict_signs(model, X_test, y_metadata, indices)

        X_pred = pick_samples_images(X_test_raw, indices)

        plot_predicitons(X_pred, signs, model_name, n_max_cols = 3)






        # Predict on web images


        








