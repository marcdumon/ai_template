# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - test.py
# md
# --------------------------------------------------------------------------------------------------------

import data_process.standard_datasets

# !/usr/bin/env python

import os
import struct
import sys

from array import array
from os import path

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

x = {_class: i for i, _class in enumerate(classes)}
print(x)


