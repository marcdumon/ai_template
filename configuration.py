# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - configuration.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Manages the configuration of the environemet and the recipe for running the machine.
It reads a yaml files into dataclasses.
"""
from dataclasses import dataclass, field
from time import sleep
from typing import List

import yaml

from my_tools.python_tools import now_str


@dataclass
class Config:
    """Dataclass with all configuration parameters."""

    device = 'cuda'
    default_config_file: str = './default_config.yml'
    default_recipe_file: str = './default_recipe.yml'
    temp_report_path: str = '../temp_reports/'
    tb_path = '../tensorboard/'
    datasets_path: str = '/media/md/Development/0_Datasets/0_standard_datasets/'

    show_batch_images = True
    show_top_losses = True
    tb_projector = True
    log_pr_curve = True
    lr_scheduler = False
    early_stopping = True
    save_last_checkpoint = True
    save_best_checkpoint = True
    creation_time: str = now_str('yyyymmdd_hhmmss')

    @staticmethod
    def save_default_yaml():
        default_rcp = Config().save_yaml(cfg.default_config_file)

    def save_yaml(self, file=None):
        if file is None: file = f'{rcp.base_path}cfg.yml'
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=None):
        """Load the recipe yaml and returns a Recipe dataclass"""
        if file is None: file = f'{rcp.base_path}cfg.yml'
        try:
            with open(file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            cfg.__dict__ = config
            return cfg
        except FileNotFoundError:
            print("Config file doesn't exist.")


cfg = Config()


@dataclass()
class Recipe:  # Prescription, Ingredient, ModusOperandi
    """
    A dataclass with all the parameters that might vary from one experiment to the other or from one stage of an experiment
    to the other stage
    """
    experiment: str = ''
    description = ''
    stage: int = 0
    seed: int = 42

    bs: int = 8 * 64
    lr: float = 3e-3
    lr_frac: List[int] = field(default_factory=lambda: [1, 1])  # By how much the lr will be devided
    max_epochs = 25
    shuffle_batch: bool = True

    # Transforms
    transforms: dict = field(default_factory=lambda: {'topilimage': True,
                                                      'randomrotation': None,
                                                      'resize': None,
                                                      'randomverticalflip': None,
                                                      'randomhorizontalflip': None,
                                                      'totensor': True,
                                                      'normalize': {
                                                          'mean': [0.1307, ],
                                                          'std': [0.3081, ]}
                                                      })

    creation_time: str = now_str('yyyymmdd_hhmmss')

    @property
    def base_path(self):
        return f'{cfg.temp_report_path}{self.experiment}/{self.creation_time}/'

    @property
    def models_path(self):
        return f'{self.base_path}models/'

    @property
    def src_path(self):
        return f'{self.base_path}src/'

    @property
    def tb_log_path(self):
        return f'{cfg.tb_path}{self.experiment}/{self.creation_time}/'

    @property
    def results_path(self):
        return f'{self.base_path}/results/'

    @staticmethod
    def save_default_yaml():
        default_rcp = Recipe().save_yaml(cfg.default_recipe_file)

    def save_yaml(self, file=None):
        if file is None:
            file = f'{self.base_path}rcp_{rcp.stage}.yml'
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=None):
        """Load the recipe yaml and returns a Recipe dataclass"""
        if file is None:
            file = f'{cls.base_path}rcp_{rcp.stage}.yml'
        try:
            with open(file, 'r') as f:
                recipe = yaml.load(f, Loader=yaml.FullLoader)
            rcp.__dict__ = recipe
            return rcp
        except FileNotFoundError:
            print("Recipe file doesn't exist.")
            raise


rcp = Recipe()

if __name__ == '__main__':
    pass
