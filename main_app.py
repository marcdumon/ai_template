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

from configuration import rcp, cfg
from data_process import MNIST_Dataset, standard_datasets
from machine import Model, run_training, setup_experiment, close_experiment
from my_tools import python_tools
from my_tools.lr_finder import LRFinder
from my_tools.python_tools import set_random_seed
from my_tools.pytorch_tools import random_split_train_valid, set_lr, summary, set_requires_grad
import numpy as np

set_random_seed(rcp.seed)

# DATA
mnist_ds = MNIST_Dataset(sample=False)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
# Model
model = Model().to(cfg.device)  # Model should be on gpu before putting parametyers in optimizer

rcp.experiment = 'baseline_all_samples'
rcp.description = 'train a model with all samples till plateau'
close_experiment(rcp.experiment, '20200104_141548')
setup_experiment()

rcp.stage = 1
rcp.max_epochs = 500
rcp.lr = 5e-3
rcp.lr_frac = [1, 10]
rcp.bs = 32
set_requires_grad(model, 'all', True, f'{rcp.models_path}requires_grad_{rcp.stage}.txt')
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()

run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

# # Data
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomRotation(90),
#     # transforms.Resize(10),
#     # transforms.RandomVerticalFlip(.5),
#     # transforms.RandomHorizontalFlip(.5),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# train.transform, valid.transform = transform, transform
# train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
# valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
# lr_finder = LRFinder(model, optimizer, loss, 'cuda')
# lr_finder.range_test(train_loader=train_loader, val_loader=valid_loader, end_lr=1e-1, num_iter=100)
#
# lr_finder.plot()
# lr_finder.reset()
# 1/0
# summary(model, (1, 28, 28), batch_size=rcp.bs, device=cfg.device, to_file=f'{rcp.models_path}summary_{rcp.stage}.txt')
# run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

# for i in range(1, 50):
#     rcp.stage = i
#     # train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
#     s = np.random.randint(100, len(train) - 100)
#     e = s + 100
#     print(s, e)
#     train_100 = deepcopy(train)
#     train_100.data = train_100.data[s: e]
#     train_100.targets = train_100.targets[s: e]
#     run_training(model, train=train_100, valid=valid, optimizer=optimizer, loss=loss)

if __name__ == '__main__':
    pass
