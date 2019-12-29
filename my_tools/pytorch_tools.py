# --------------------------------------------------------------------------------------------------------
# 2019/12/29
# src - pytorch_tools.py
# md
# --------------------------------------------------------------------------------------------------------
from copy import deepcopy

import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

from my_tools.python_tools import print_file


# MODELS
def set_requires_grad(model, names, requires_grad):
    """
    Set the requires_grad on parameters from model based on name

    Args:
        model: model containing parameters that need to set requires_grad
        names: (str, int of list[str]) Parameter name or part of a parameter name.
            If part, then all parameters with that part in their name will have requires_grad set.
        requires_grad: (bool) Sets the requires_grad of the parameter
    """
    if isinstance(names, str): names = [names]
    if isinstance(names, int): names = [str(names)]
    print_file(f'Setting requires_grad:')
    for n in names:
        for name, param in [(name, param) for name, param in model.named_parameters() if n in name]:
            param.requires_grad = requires_grad
            print_file(f'\t{name:40}: {requires_grad}')


def set_lr(model, names, lr):
    """
    Sets the learning rate for parameters defined by name to be used in optimizers
    Args:
        model: model containing parameters that need to set requires_grad
        names: (str, int of list[str]) Parameter name or part of a parameter name.
            If part, then all parameters with that part in their name will have lr set.
        lr: learning rate for the parameter(s)

    Returns:
        List of dictionaries of the form [{'params': param, 'lr': lr}, ...] to be used in optimizer
    """
    if isinstance(names, str): names = [names]
    if isinstance(names, int): names = [str(names)]
    params = []
    for name in names:
        params += [{'name': n, 'params': p, 'lr': lr} for n, p in model.named_parameters() if name in n]  # name added for debugging
    return params


# DATA
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


def get_mean_and_std(dataset):  # Todo: check this out
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
