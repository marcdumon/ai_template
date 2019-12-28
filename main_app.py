# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - main_app.py
# md
# --------------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
import torch as th

from configuration import rcp
from data_process import MNIST_Dataset
from data_process.utils import random_split_train_valid
from machine import xxx, Model
from test3 import run_rcp

# DATA
mnist_ds = MNIST_Dataset(sample=False)
train, valid = random_split_train_valid(dataset=mnist_ds, valid_frac=.2)
train.data = train.data[:50]
train.targets = train.targets[:50]
print(len(train), len(valid))

'''
How to structure this?
- usecase1: finetune witch certain layers frozen
- usecase2: stage1: train with fc unfrozen till plateau, unfreeze layer4, train till plateau, unfreeze layer3, ... 
rcp.stage=1
unfreeze fc
train
rcp.stage=2
unfreeze L4
...
# check if setting lr=0 for a layer is the same as freezing it?

ALso: 
 you want to use this with a scheduler (e.g. CyclicLR), then the lr’s of the parameter groups should also be passed to 
 the scheduler constructor as a list of floats. Otherwise, the group dependent lr’s are lost.
 
 
 
 
'''

# Model
model=Model()
optimizer = th.optim.Adam(model.parameters(), lr=3e-3)
# optimizer = th.optim.Adagrad(model.parameters(), lr=3e-3)
loss = th.nn.NLLLoss()

trainer = xxx(train=train, valid=valid, optimizer='', loss='')
print(trainer.state)
trainer.run(max_epochs=rcp.max_epochs)

if __name__ == '__main__':
    pass
