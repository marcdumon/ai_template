# --------------------------------------------------------------------------------------------------------
# 2019/12/27
# src - python_tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
from math import sqrt, ceil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch as th


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def now_str(pattern='yyyymmdd_hhmmss'):
    """
    The currect datetime according to pattern
    Args:
        pattern: string indicating the format of the returned datetime.
    Return:
        datetime string in the format
    """
    # Todo: make more generic
    # Todo: chack print(f'{now:%Y-%m-%d %H:%M}')
    '''
    Using the following:
    char_counts = {}
    for c in pattern:
        char_counts[c] = char_counts.get(c, 0) + 1 # if c not a key in char_counts, create it with value0, otherwise add 1

    '''

    now = datetime.now()

    if pattern == 'yyyymmdd_hhmmss': return f'{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}'

    return 'Pattern not implemented!'


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


def create_path(path: str):
    """
    Creates a path if it doesn't already exists
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def copy_file():
    pass


if __name__ == '__main__':
    print(now_str())
    create_path('./xxx')
