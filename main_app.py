# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
import numpy as np
import torch as th

from configuration import rcp
from data_process import MNIST_Dataset
from machine import Model, run_training
from my_tools.python_tools import set_random_seed
from my_tools.pytorch_tools import random_split_train_valid, set_lr, summary, set_requires_grad

set_random_seed(rcp.seed)

# DATA
mnist_ds = MNIST_Dataset(sample=True)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
# train.data = train.data[:50]
# train.targets = train.targets[:50]
print(len(train), len(valid))
print(type(train[0][0]))
# Model
model = Model()
set_requires_grad(model, 'all', True)
params = set_lr(model, ['conv', 'fc'], 1e-3)
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()
summary(model, np.swapaxes(train[0][0], 0, 2).shape, batch_size=rcp.bs, device='cpu', to_file=rcp.summary_file)
run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

if __name__ == '__main__':
    pass
