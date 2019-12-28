# --------------------------------------------------------------------------------------------------------
# 2019/12/27
# src - tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
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
    '''
    Using the following:
    char_counts = {}
    for c in pattern:
        char_counts[c] = char_counts.get(c, 0) + 1 # if c not a key in char_counts, create it with value0, otherwise add 1

    '''

    now = datetime.now()

    if pattern == 'yyyymmdd_hhmmss': return f'{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}'

    return 'Pattern not implemented!'


if __name__ == '__main__':
    print(now_str())
