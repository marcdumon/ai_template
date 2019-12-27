# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - test.py
# md
# --------------------------------------------------------------------------------------------------------
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.transforms import transforms
from tqdm import tqdm

# from data_process.standard_datasets import MNIST_Dataset

# !/usr/bin/env python

import os
import struct
import sys

from array import array
from os import path
import numpy as np
import random
import torch as th

from data_process import MNIST_Dataset
from data_process.utils import stratified_split_train_valid, random_split_train_valid
from machine import xxx, Model
from models.standard_models import MNSIT_Simple
from my_tools.tools import set_random_seed
from visualization.utils import show_mpl_grid

seed = 42
set_random_seed(seed)

bs = 8
expertiment_name = 'XXX'
tb_logdir = '/media/md/Development/My_Projects/0_ml_project_template.v1/tensorboard/' + expertiment_name

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# MNIST_Dataset.create_samples(250)
mnist_ds = MNIST_Dataset(sample=False, transform=transform)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
train.data=train.data[:50]
train.targets=train.targets[:50]
print(len(train), len(valid))
train_loader = DataLoader(train, batch_size=bs, num_workers=8, shuffle=True)
val_loader = DataLoader(valid, batch_size=bs, num_workers=8, shuffle=True)

model = Model().cuda()
# optimizer = th.optim.Adam(model.parameters(), lr=3e-3)
optimizer = th.optim.Adagrad(model.parameters(), lr=3e-3)
loss = th.nn.NLLLoss()
trainer = xxx(model, train_loader, val_loader, optimizer, loss)
trainer.run(train_loader, max_epochs=100)

'''
Failed to save model graph: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
ITERATION - loss: 0.58: 100%|██████████| 10/10 [00:19<00:00,  1.15s/it]Train Results - Epoch: 1  Avg accuracy: 0.88231 Avg loss: 0.39716
Valid Results - Epoch: 1  Avg accuracy: 0.88550 Avg loss: 0.39860
ITERATION - loss: 0.37: 20it [00:36,  1.32it/s]Train Results - Epoch: 2  Avg accuracy: 0.92717 Avg loss: 0.24810
Valid Results - Epoch: 2  Avg accuracy: 0.92567 Avg loss: 0.25779
'''