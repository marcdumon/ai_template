# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
import torch as th

from configuration import rcp
from data_process import MNIST_Dataset
from machine import Model, run_training
from my_tools.python_tools import set_random_seed
from my_tools.pytorch_tools import random_split_train_valid, set_lr
from visualization.torchsummary import summary

set_random_seed(rcp.seed)

# DATA
mnist_ds = MNIST_Dataset(sample=True)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
# train.data = train.data[:50]
# train.targets = train.targets[:50]
print(len(train), len(valid))

# Model
model = Model()

for name, param in model.named_parameters():
    print('====>', name, '<===')
summary(model, (1, 28, 28), batch_size=1, device='cpu', to_file=None)

params = set_lr(model, 'conv', 3e-4)
params += set_lr(model, 'fc', 3e-2)

optimizer = th.optim.Adam(params=params, lr=1e999)
print(optimizer)
loss = th.nn.NLLLoss()

run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

if __name__ == '__main__':
    pass
