# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - utils.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of tools to manipulate data
"""
from torch.utils.data import random_split


def random_split_train_valid(dataset, valid_frac):
    """
    Randomly splits a dataset into a train and valid dataset subset.

    Args:
        dataset: The dataset to randomly split into a train and valid dataset subset.
        valid_frac: Between 0 and 1. The fraction of the dataset that will be randomly split into a valid dataset subset.

    Return:
        tuple: (train_ds, valid_ds) where train_ds is the train dataset subset and valid_ds is the valid dataset subset.
    """
    assert 0 < valid_frac < 1, "valid_frac must be bigger than 0 and smaller than 1"
    train_size = round(len(dataset) * (1 - valid_frac))
    valid_size = len(dataset) - train_size

    train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
    return train_ds, valid_ds


if __name__ == '__main__':
    pass
