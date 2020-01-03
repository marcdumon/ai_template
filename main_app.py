# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
import torch as th
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from configuration import rcp, cfg
from data_process import MNIST_Dataset
from machine import Model, run_training, setup_experiment, close_experiment
from my_tools.lr_finder import LRFinder
from my_tools.python_tools import set_random_seed
from my_tools.pytorch_tools import random_split_train_valid, set_lr, summary, set_requires_grad
import numpy as np

close_experiment('baseline', '20200103_230758')
setup_experiment()
set_random_seed(rcp.seed)
# DATA
mnist_ds = MNIST_Dataset(sample=False)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
train.data = train.data[:124]
train.targets = train.targets[:124]
# train.targets=np.random.permutation(train.targets)
# targets = []
# for i in range(len(valid.targets)):
#     if valid.targets[i] in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         targets.append(valid.targets[i])
#     else:
#         targets.append(np.random.randint(0, 1))
#
# valid.targets = targets

# Model
model = Model().to(cfg.device)  # Model should be on gpu before putting parametyers in optimizer

# Stage 1
rcp.stage = 1
rcp.max_epochs = 100
rcp.lr = 3e-3
rcp.lr_frac = [1, 77]
rcp.bs = 32
set_requires_grad(model, 'all', True, f'{rcp.models_path}requires_grad_{rcp.stage}.txt')
params = set_lr(model, ['fc1', 'fc2'], rcp.lr / rcp.lr_frac[0])
params += set_lr(model, ['conv1', 'conv2'], rcp.lr / rcp.lr_frac[1])
optimizer = th.optim.Adam(params=params, lr=1e999)
loss = th.nn.NLLLoss()
# Todo: Move this to a function???
# Data
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomRotation(90),
    # transforms.Resize(10),
    # transforms.RandomVerticalFlip(.5),
    # transforms.RandomHorizontalFlip(.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# train.transform, valid.transform = transform, transform
# train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
# valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
# lr_finder = LRFinder(model, optimizer, loss, 'cuda')
# lr_finder.range_test(train_loader=train_loader, val_loader=valid_loader, end_lr=1e-1, num_iter=100)
#
# lr_finder.plot()
# lr_finder.reset()

summary(model, (1, 28, 28), batch_size=rcp.bs, device=cfg.device, to_file=f'{rcp.models_path}summary_{rcp.stage}.txt')
run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

for i in range(2, 20):
    rcp.stage = i
    train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
    s = np.random.randint(124, len(train) - 124)
    e = s + 124
    print(s, e)
    train.data = train.data[s: e]
    train.targets = train.targets[s: e]
    run_training(model, train=train, valid=valid, optimizer=optimizer, loss=loss)

if __name__ == '__main__':
    pass
