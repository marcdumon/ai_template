# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - utils.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of tools to manipulate data
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


def random_split_train_valid(dataset, valid_frac):
    """
    Randomly splits a dataset into a train and valid dataset subset.
    Important: Random split doesn't guarantee that the train and valid dataset subsets have the same the class distributions.

    Args:
        dataset: The dataset to split into a train and valid dataset subset.
        valid_frac: Between 0 and 1. The fraction of the dataset that will be split into a valid dataset subset.

    Return:
        tuple: (train_ds, valid_ds) where train_ds is the train dataset subset and valid_ds is the valid dataset subset.
    """
    assert 0 < valid_frac < 1, "valid_frac must be bigger than 0 and smaller than 1"
    data = dataset.data
    targets = dataset.targets
    train, valid = deepcopy(dataset), deepcopy(dataset)
    train.data, valid.data, train.targets, valid.targets = train_test_split(data, targets, test_size=valid_frac, stratify=None)
    return train, valid


def stratified_split_train_valid(dataset, valid_frac):
    """
    Split a dataset into a train and valid dataset subset using stratification. Stratification means that the split will try
    to approximate the class distributions from the dataset to the train and valid datasets.

    Args:
        dataset: The dataset to split into a train and valid dataset subset.
        valid_frac: Between 0 and 1. The fraction of the dataset that will be split into a valid dataset subset.

    Return:
        tuple: (train_ds, valid_ds) where train_ds is the train dataset subset and valid_ds is the valid dataset subset.
    """
    assert 0 < valid_frac < 1, "valid_frac must be bigger than 0 and smaller than 1"
    data = dataset.data
    targets = dataset.targets
    train, valid = deepcopy(dataset), deepcopy(dataset)
    train.data, valid.data, train.targets, valid.targets = train_test_split(data, targets, test_size=valid_frac, stratify=targets)
    return train, valid


def get_class_distribution(dataset):
    """
    Calculates the class distribution, ie the number of samples per class.

    Args:
        dataset: a data_process.standard_datasets dataset that implemented the classes attibute

    Return:
        pandas.DataFrame with index=class index and columns=['class', 'n_samples'] sorted by index
    """
    classes = dataset.classes
    index = dataset.targets
    class_distribution = pd.DataFrame(columns=['class', 'n_samples', 'normalised'])
    class_distribution['n_samples'] = pd.Series(index).value_counts().sort_index()
    class_distribution['normalised'] = class_distribution['n_samples'] / max(class_distribution['n_samples'])
    class_distribution['class'] = classes
    return class_distribution


def get_mean_and_std(dataset): # Todo: check this out
    """ Compute the mean and std value of dataset. From: https://github.com/isaykatsman/pytorch-cifar/blob/master/utils.py """
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = th.zeros(3)
    std = th.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


if __name__ == '__main__':
    pass
