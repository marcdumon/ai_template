# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - test.py
# md
# --------------------------------------------------------------------------------------------------------
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.transforms import transforms
from tqdm import tqdm

from data_process.standard_datasets import MNIST_Dataset

# !/usr/bin/env python

import os
import struct
import sys

from array import array
from os import path
import numpy as np
import random
import torch as th

from data_process.utils import stratified_split_train_valid, random_split_train_valid
from visualization.utils import show_mpl_grid

seed = 42
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
# th.backends.cudnn.deterministic = True
# th.backends.cudnn.benchmark = False
bs = 8 * 512

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
mnist_ds = MNIST_Dataset(sample=False, transform=transform)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
print(len(train), len(valid))

train_loader = DataLoader(train, batch_size=bs, num_workers=8, shuffle=True)
val_loader = DataLoader(valid, batch_size=bs, num_workers=8, shuffle=True)
one_valid_batch = next(iter(val_loader))  # => list []

# images = list(one_valid_batch[0].numpy())
# targets = list(one_valid_batch[1].numpy())
# show_mpl_grid(images, targets)

model = resnet18(pretrained=True, progress=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

model.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
# optimizer = th.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
optimizer = th.optim.Adam(model.fc.parameters(), lr=1e-4)
loss = th.nn.NLLLoss()

trainer = create_supervised_trainer(model, optimizer, loss, device='cuda')
evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'nll': Loss(loss)}, device='cuda')

desc = "ITERATION - loss: {:.2f}"
pbar = tqdm(initial=0, leave=False, total=len(train_loader),
            desc=desc.format(0))


@trainer.on(Events.ITERATION_COMPLETED(every=1))
def log_training_loss(engine):
    pbar.desc = desc.format(engine.state.output)
    pbar.update(1)


# def log_training_loss(trainer):
# print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    pbar.refresh()
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write("\nTrain Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
               .format(engine.state.epoch, avg_accuracy, avg_nll)               )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Valid Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))


trainer.run(train_loader, max_epochs=100)
