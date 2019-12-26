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
from visualization.utils import show_mpl_grid
import configuration



seed = 42
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
bs = 8 * 64
tb_logdir = '/media/md/Development/My_Projects/0_ml_project_template.v1/tensorboard/'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
mnist_ds = MNIST_Dataset(sample=False, transform=transform)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)

train_loader = DataLoader(train, batch_size=bs, num_workers=8, shuffle=True)
val_loader = DataLoader(valid, batch_size=bs, num_workers=8, shuffle=True)
one_valid_batch = next(iter(val_loader))  # => list []

# images = list(one_valid_batch[0].numpy())
# targets = list(one_valid_batch[1].numpy())
# show_mpl_grid(images, targets)


# model = resnet18(pretrained=True, progress=True)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
# model = MNSIT_Simple()
model = Model()

# optimizer = th.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
# optimizer = th.optim.Adam(model.fc.parameters(), lr=1e-4)
optimizer = th.optim.Adam(model.parameters(), lr=3e-3)
loss = th.nn.NLLLoss()

trainer = xxx(model, train_loader, val_loader, optimizer, loss)
trainer.run(train_loader, max_epochs=100)
