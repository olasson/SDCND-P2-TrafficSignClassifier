"""
This file contains function(s) for visualizing data.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

from code.misc import pick_samples_labels, distribution_is_uniform

N_IMAGES_MAX = 50
N_DISTRIBUTIONS_MAX = 3

def plot_images(X, 
            titles_top = None, 
            titles_bottom = None, 
            title_fig_window = None, 
            fig_size = (15, 15), 
            font_size = 10, 
            cmap = None, 
            n_max_cols = 5, 
            titles_bottom_h_align = 'center', 
            titles_bottom_v_align = 'top', 
            titles_bottom_pos = (16, 32)):

    n_images = len(X)

    if n_images > N_IMAGES_MAX:
        print("ERROR: code.plot.plot_images(): You're trying to show", n_images, "images. Max number of allowed images:", N_IMAGES_MAX)
        return

    n_cols = int(min(n_images, n_max_cols))
    n_rows = int(np.ceil(n_images / n_cols))

    fig = plt.figure(title_fig_window, figsize = fig_size)

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)

        plt.imshow(X[i].astype('uint8'), cmap = cmap)

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
    plt.show()

def plot_distributions(distributions, y_metadata = None, title = None, fig_size = (15, 10), font_size = 6):

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


    plt.figure(figsize = fig_size)

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, distribution in enumerate(distributions):
        _, classes_count = np.unique(distribution, return_counts = True)
        classes_count, _ = pick_samples_labels(classes_count, classes_order)
        plt.barh(classes, classes_count, color = colors[i])

    plt.yticks(classes, y_ticks, fontsize = font_size)

    plt.xlabel("Number of each class", fontsize = font_size + 14)

    plt.title(title, fontsize = font_size + 20)

    if y_metadata is None:
        plt.ylabel("Class ID", fontsize = font_size + 14)

    plt.show()

