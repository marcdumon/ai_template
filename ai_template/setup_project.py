# --------------------------------------------------------------------------------------------------------
# 2020/01/10
# src - setup_project.py
# md
# --------------------------------------------------------------------------------------------------------
import os
import shutil
from pathlib import Path


def setup_project():
    """
        Create directories for project:
        ./experiments
        ./notebooks
        ./src
        ./temp_experiments
        ./tensorboard
        Copy main_app.py to ./
    """
    for path in ['./experiments', './notebooks', './src', './temp_experiments', './tensorboard']:
        Path(path).mkdir(exist_ok=True)
    script_path = os.path.dirname(os.path.realpath(__file__))
