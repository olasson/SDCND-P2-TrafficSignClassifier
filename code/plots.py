"""
This file contains function(s) for visualizing data.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as path_join

from code.misc import pick_samples_1D, distribution_is_uniform

N_IMAGES_MAX = 50
N_DISTRIBUTIONS_MAX = 3

def plot_images(images, 
            titles_top = None, 
            titles_bottom = None, 
            title_fig_window = None, 
            fig_size = (15, 15), 
            font_size = 10, 
            cmap = None, 
            n_max_cols = 5, 
            titles_bottom_h_align = 'center', 
            titles_bottom_v_align = 'top', 
            titles_bottom_pos = (16, 32),
            file_path_save = None):

    """
    Show a set of images.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images, RGB or grayscale.
    titles_top: (None | list)
        A set of image titles to be displayed on top of an image.
    titles_bottom: (None | list)
        A set of image titles to be displayed at the bottom of an image.
    title_fig_window: (None | str)
        Title for the figure window.
    figsize: (int, int)
        Tuple specifying figure width and height in inches.
    fontsize: int
        Fontsize of 'titles_top' and 'titles_bottom'.
    cmap: (None | str)
        RGB or grayscale.
    n_max_cols: int
        Maximum number of columns allowed in figure.
    titles_bottom_h_align: str
        Horizontal alignment of 'titles_bottom'.
    titles_bottom_v_align: str
        Vertical alignment of 'titles_bottom'.
    titles_bottom_pos: (int, int)
        Tuple containing the position of 'titles_bottom'.
    titles_bottom_transform: str
        Coordinate system used by matplotlib for 'titles_bottom'.
    file_path_save: (None | str)
        File path specifying where the figure should be saved.

    Outputs
    -------
    plt.figure
        Figure showing 'images' in an (n_rows x n_cols) layout.
    
    """

    n_images = len(images)

    if n_images > N_IMAGES_MAX:
        print("ERROR: code.plot.plot_images(): You're trying to show", n_images, "images. Max number of allowed images:", N_IMAGES_MAX)
        return

    n_cols = int(min(n_images, n_max_cols))
    n_rows = int(np.ceil(n_images / n_cols))

    fig = plt.figure(title_fig_window, figsize = fig_size)

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)

        plt.imshow(images[i].astype('uint8'), cmap = cmap)

        plt.xticks([])
        plt.yticks([])

        if titles_top is not None:
            plt.title(titles_top[i], fontsize = font_size)

        if titles_bottom is not None:
            plt.text(titles_bottom_pos[0], titles_bottom_pos[1], 
                     titles_bottom[i],
                     verticalalignment = titles_bottom_v_align, 
                     horizontalalignment = titles_bottom_h_align,
                     fontsize = font_size - 3)

    plt.tight_layout()

    if file_path_save is not None:
        fig_mgr = plt.get_current_fig_manager()
        fig_mgr.window.showMaximized()
        plt.show()
        fig.savefig(file_path_save)
        plt.close()
    else:
        plt.show()


def plot_predictions(images, signs, title_fig_window = None, n_max_cols = 3, file_path_save = None):
    """
    Show a set of model predictions.
    
    Inputs
    ----------
    images: numpy.ndarray
        Numpy array containing a set of images, RGB or grayscale.
    signs: numpy.ndarray
        Numpy array containing a set of model predictions, corresponding to 'images'.
    titles_bottom: (None | list)
        A set of image titles to be displayed at the bottom of an image
    title_fig_window: (None | str)
        Title for the figure window
    n_max_cols: int
        Maximum number of columns allowed in figure
    file_path_save: (None | str)
        File path specifying where the figure should be saved

    Outputs
    -------
    plt.figure
        Figure showing 'images' in an (n_rows x n_cols) layout.

    Notes
    -------
        See code.models.predict_signs().
    
    """

    plot_images(images, titles_bottom = signs, title_fig_window = title_fig_window, 
                font_size = 12, n_max_cols = n_max_cols, titles_bottom_h_align = 'left', titles_bottom_pos = (34, 7.0),
                file_path_save = file_path_save)

def plot_distributions(distributions, y_metadata = None, title = None, title_fig_window = None, fig_size = (15, 10), font_size = 6):
    """
    Show label distribution
    
    Inputs
    ----------
    distributions: numpy.ndarray
        Numpy array containing up to N_DISTRIBUTIONS_MAX set of labels - '[labels1, ... labelsN]'
    title: (None | list)
        A title for the figure.
    title_fig_window: (None | str)
        Title for the figure window.
    figsize: (int, int)
        Tuple specifying figure width and height in inches.
    fontsize: int
        Fontsize of 'title'
        
    Outputs
    -------
    plt.figure
        Figure showing label class distribution(s)
    
    """

    def _remove_none(distributions):
        res = []
        for distribution in distributions:
            if distribution is not None:
                res.append(distribution)
        return res

    n_distributions = len(distributions)

    if n_distributions > N_DISTRIBUTIONS_MAX:
        print("ERROR: code.plot.plot_distributions(): You're trying to show", n_distributions, "images. Max number of allowed images:", N_DISTRIBUTIONS_MAX)
        return

    distributions = _remove_none(distributions)

    distributions = sorted(distributions, key=len, reverse=True)
    
    order_index = 0

    if distribution_is_uniform(distributions[0]):
        if len(distributions) > 1:
            order_index = 1
        elif len(distributions) > 2:
            order_index = 2

    classes, classes_count_order = np.unique(distributions[order_index], return_counts = True)

    # To make the plot tidy: 
    # ensure that the order of classes fits a reverse sort of 'classes_count_order'
    # 'classes_order' then determines the order of classes on the y-axis
    classes_order = [tmp for _,tmp in sorted(zip(classes_count_order, classes), reverse = True)]

    n_classes = len(classes)
    
    if y_metadata is not None:
        y_ticks = np.empty((n_classes), dtype = 'U25')
        for i in range(n_classes):
            y_ticks[i] = y_metadata[classes_order[i]]
    else:
        y_ticks = classes_order


    plt.figure(title_fig_window, figsize = fig_size)

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, distribution in enumerate(distributions):
        _, classes_count = np.unique(distribution, return_counts = True)
        classes_count = pick_samples_1D(classes_count, classes_order, dtype = np.int)
        plt.barh(classes, classes_count, color = colors[i])

    plt.yticks(classes, y_ticks, fontsize = font_size)

    plt.xlabel("Number of each class", fontsize = font_size + 14)

    plt.title(title, fontsize = font_size + 20)

    if y_metadata is None:
        plt.ylabel("Class ID", fontsize = font_size + 14)

    plt.show()


def plot_model_history(history, model_name = None, lrn_rate = None, batch_size = None, max_epochs = None, file_path_save = None,):
    """
    Plot model history and metadata
    
    Inputs
    ----------
    model_name: string
        Name of the model
    history: Keras History Object
        Model history (output from .fit)
    path_save: (None | string)
        Path to where the plot will be saved. 
    lrn_rate: (None | float)
        Model learning rate
    batch_size: (None | int)
        Model batch size
    max_epochs: (None | int)
        Model max epochs 
        
    Outputs
    -------
    plt.figure
        Figure showing model history and metadata, either shown directly or saved in location 'path_save'
    
    """

    if model_name is None:
        model_name = 'model'

    train_log = history.history['loss']
    valid_log = history.history['val_loss']
    
    train_loss = train_log[-1]
    valid_loss = valid_log[-1]
    
    text = "Training/Validation Loss: " + str(round(train_loss, 3)) + '/' + str(round(valid_loss, 3))   
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    c1 = colors[0]
    c2 = colors[1]
    
    fig, ax1 = plt.subplots(figsize = (9, 6))
    
    ax1.set_xlabel('Epochs')    
    ax1.set_ylabel('Loss')

    x = np.arange(1, len(train_log) + 1)
    
    ax1.plot(x, train_log, label = 'Train Loss', color = c1)
    ax1.plot(x, valid_log, label = 'Validation Loss', color = c2)


    stopping_epoch = len(history.history['loss'])

    # ---------- Construct a title for the plot ---------- # 

    model_name_title = 'Model Name: '+ model_name + ' | '

    if lrn_rate is not None:
        lrn_rate_title = 'Lrn rate: ' + str(lrn_rate) + ' | '
    else:
        lrn_rate_title = ''

    if batch_size is not None:
        batch_size_title = 'Batch size: ' + str(batch_size) + ' | '
    else:
        batch_size_title = ''

    if max_epochs is not None:
        epochs_title = 'Stopp/Max (Epoch): ' + str(stopping_epoch) + '/' + str(max_epochs)
    else:
        epochs_title = 'Stopp Epoch: ' + str(stopping_epoch)

    plt.title(model_name_title + lrn_rate_title + batch_size_title + epochs_title)

    # ---------- Misc ---------- #
    
    fig.text(0.5, 0, text,
                verticalalignment = 'top', 
                horizontalalignment = 'center',
                color = 'black', fontsize = 10)
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = (0.7, 0.5))
    fig.tight_layout()

    # ---------- Show or save ---------- #
    
    # If the user has opted to save the model history, don't show the plot directly
    if file_path_save is not None:
        fig.savefig(file_path_save, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
