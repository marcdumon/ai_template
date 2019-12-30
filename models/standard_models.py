# --------------------------------------------------------------------------------------------------------
# 2019/12/25
# src - standard_models.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of Pytorch models
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Todo: Put the model parameters
# checkout:
'''
class MnistResNet(ResNet):
    def __init__(self):
   ---> super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
    ===> return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)
from: https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/ 
But bigg mistake in loss (see: https://github.com/marrrcin/pytorch-resnet-mnist/issues/1)
'''


# checkout: https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/


class MNSIT_Simple(nn.Module):

    def __init__(self):
        super(MNSIT_Simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 32x26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 64x24x24
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 128
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)  # 10

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = th.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


'''
torch.Size([2, 32, 26, 26])
torch.Size([2, 64, 24, 24])
torch.Size([2, 64, 12, 12])
torch.Size([2, 64, 12, 12])
torch.Size([2, 9216])
torch.Size([2, 128])
torch.Size([2, 128])
torch.Size([2, 10])
torch.Size([2, 10])
'''
