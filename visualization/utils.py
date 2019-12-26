# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - utils.py
# md
# --------------------------------------------------------------------------------------------------------
from math import floor, ceil, sqrt

import matplotlib.pyplot as plt


def show_mpl_grid(images, titles=None, figsize=(10, 7), shape=(0, 0), cm='gray'):
    """
    Shows images in a grid. Uses matplotlib pyplot

    Args:
        images: list of images to show in a grid
        titles: list of titles for each image
        figsize: (horizontal, vertical) a tuple passing to plt figuresize
        shape: (rows, columns) the shape of the grid. The shape will be automatically calculated when it's not provided
        cm: matplotlib cmap

    Returns:
        Shows a matplotlib grig of images
    """
    # Todo: have this accept a list, np.array or tensor of images. Have different functions for that?
    # Todo: change name to mpl_show_grid? mlp_show_batch?
    if shape == (0, 0):
        l = len(images)
        r = int(sqrt(l))
        c = int(ceil(l / r))
        shape = (r, c)

    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = plt.subplot(shape[0], shape[1], 1 + i)
        ax.imshow(images[i], cmap=cm)
        if titles:
            ax.title.set_text(titles[i])
    # plt.tight_layout(1.08)
    plt.show(block=False)
    # plt.pause(2)
    plt.waitforbuttonpress(0)
    plt.close()

