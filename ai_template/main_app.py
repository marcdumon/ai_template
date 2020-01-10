# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
import random
from copy import deepcopy
from time import sleep

import torch as th
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from .configuration import rcp, cfg
from .data_process import MNIST_Dataset, standard_datasets
from .data_process.standard_datasets import FashionMNIST_Dataset
from .machine import Model, run_training, setup_experiment, close_experiment
from .my_tools import python_tools
from .my_tools.lr_finder import LRFinder
from .my_tools.python_tools import set_random_seed, print_file
from .my_tools.pytorch_tools import random_split_train_valid, set_lr, summary, set_requires_grad
import numpy as np

set_random_seed(rcp.seed)

rcp.experiment = 'normalization_or_not_stability_check'
rcp.description = 'Stage 10 till 19 is with normalization, stage 20 till 29 without. Within each stage, the seed varies from 1 till 10 '
rcp.stage = 1
rcp.max_epochs = 100
rcp.lr = 1e-3
rcp.lr_frac = [1, 1]
rcp.bs = 32
lr_find = False

# close_experiment(rcp.experiment, '20200105_003914')
setup_experiment()

# DATA
mnist_ds = FashionMNIST_Dataset(sample=False)
# mnist_ds = MNIST_Dataset(sample=False)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)

# Model
model = Model().to(cfg.device)  # Model should be on gpu before putting parametyers in optimizer

set_requires_grad(model, 'all', True, f'{rcp.models_path}requires_grad_{rcp.stage}.txt')
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()

model = run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss, lr_find=lr_find)

print(rcp.creation_time)


if __name__ == '__main__':
    pass
