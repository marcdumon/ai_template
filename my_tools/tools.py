# --------------------------------------------------------------------------------------------------------
# 2019/12/27
# src - tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
import numpy as np
import torch as th


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
