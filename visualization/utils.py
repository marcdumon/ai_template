# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - utils.py
# md
# --------------------------------------------------------------------------------------------------------
from math import floor, ceil, sqrt

import matplotlib.pyplot as plt


def print_file(txt, file=None, append=True):
    """
    Print to console. If file given then also print to file

    Args:
        txt: text to print or to save in file.
        file: if given, txt will be saved in file
        append: if false then a new file will be created, if tre then txt will be appended to an existing file.
    """
    print(txt)
    mode = 'w'
    if append: mode = 'a'
    if file: print(txt, file=open(file, mode))


def show_mpl_grid(images, titles=None, figsize=(10, 7), gridshape=(0, 0), cm='gray'):
    # Todo: have this accept a list, np.array or tensor of images. Have different functions for that?
    # Todo: change name to mpl_show_grid? mlp_show_batch?
    """
    Shows images in a grid. Uses matplotlib pyplot

    Args:
        images: list of images to show in a grid
        titles: list of titles for each image
        figsize: (horizontal, vertical) a tuple passing to plt figuresize
        gridshape: (rows, columns) the shape of the grid. The shape will be automatically calculated when it's not provided
        cm: matplotlib cmap

    Returns:
        Shows a matplotlib grig of images
    """
    # Matplotlib needs grayscal images of shape (M,N), not (M,N,1)
    if images.shape[-1] == 1: images = images[:, :, :, 0]
    if gridshape == (0, 0):
        l = len(images)
        r = int(sqrt(l))
        c = int(ceil(l / r))
        gridshape = (r, c)
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = plt.subplot(gridshape[0], gridshape[1], 1 + i)
        ax.imshow(images[i], cmap=cm)
        if titles.any():
            ax.title.set_text(titles[i])
    # plt.tight_layout(1.08)
    plt.show(block=False)
    # plt.pause(2)
    plt.waitforbuttonpress(0)
    plt.close()
