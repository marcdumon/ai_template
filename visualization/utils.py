# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - utils.py
# md
# --------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt


def show_mpl_grid(images, titles, figsize, shape, cm='gray'):
    """
    Shows images in a grid. Uses matplotlib pyplot

    Args:
        images: list of images to show in a grid
        titles: list of titles for each image
        figsize: (horizontal, vertical) a tuple passing to plt figuresize
        shape: (rows, columns) the shape of the grid
        cm: matplotlib cmap

    Returns:
        Shows a matplotlib grig of images
    """
    fig=plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = plt.subplot(shape[0], shape[1], 1 + i)
        ax.imshow(images[i], cmap=cm)
        ax.title.set_text(titles[i])
    plt.tight_layout(1.08)
    plt.show()
