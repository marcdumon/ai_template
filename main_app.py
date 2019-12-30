# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
import numpy as np
import torch as th

from configuration import rcp, cfg
from data_process import MNIST_Dataset
from machine import Model, run_training
from my_tools.python_tools import set_random_seed
from my_tools.pytorch_tools import random_split_train_valid, set_lr, summary, set_requires_grad

set_random_seed(rcp.seed)

# DATA
mnist_ds = MNIST_Dataset(sample=False)

train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
# train.data = train.data[:50]
# train.targets = train.targets[:50]

# Model
model = Model().to('cuda') # Model should be on gpu before putting parametyers in optimizer

# Stage 1
rcp.stage = 1
rcp.max_epochs = 200
rcp.lr = 1e-3
rcp.lr_frac = [1, 1]
set_requires_grad(model, 'all', True)
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()
summary(model, (1, 28, 28), batch_size=rcp.bs, device='cuda', to_file=rcp.summary_file)
run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

# Stage 2
rcp.stage = 2
rcp.max_epochs = 2
rcp.lr = rcp.lr/10
rcp.lr_frac = [1, 1]
set_requires_grad(model, 'all', True)
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()
summary(model, (1, 28, 28), batch_size=rcp.bs, device='cuda', to_file=rcp.summary_file)
run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

# Stage 3
rcp.stage = 3
rcp.max_epochs = 2
rcp.lr = rcp.lr/10
rcp.lr_frac = [5, 1]
set_requires_grad(model, 'all', True)
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()
summary(model, (1, 28, 28), batch_size=rcp.bs, device='cuda', to_file=rcp.summary_file)
run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)


if __name__ == '__main__':
    pass
