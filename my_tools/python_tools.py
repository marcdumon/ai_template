# --------------------------------------------------------------------------------------------------------
# 2019/12/27
# src - python_tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
from pathlib import Path

import numpy as np
import torch as th
from datetime import datetime


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
