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
from typing import List

import yaml

from my_tools.python_tools import now_str


@dataclass
class Config:
    """Dataclass with all configuration parameters."""

    default_config_file: str = './default_config.yml'
    default_recipe_file: str = './default_recipe.yml'
    report_path: str = '../reports/'
    tb_path = '../tensorboard/'

    device = 'cuda'
    creation_time: str = now_str('yyyymmdd_hhmmss')

    def __post_init__(self):
        self.creation_time = now_str('yyyymmdd_hhmmss')

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
    experiment: str = 'baseline'
    stage: int = 1
    seed: int = 19640601

    bs: int = 8 * 64
    lr: float = 3e-3
    lr_frac: List[int] = field(default_factory=lambda: [1, 1])  # By how much the lr will be devided
    max_epochs = 25

    shuffle_batch: bool = True
    # model_parameters: Todo: separated, inheritated or nested. Where putting the parameters?
    # test: Config = field(default_factory=lambda: cfg)

    creation_time: str = now_str('yyyymmdd_hhmmss')

    base_path: str = f'{cfg.report_path}{experiment}/{creation_time}/'
    models_path: str = f'{base_path}models/'
    results_path: str = f'{base_path}/results/'
    src_path: str = f'{base_path}src/'
    tb_log_path: str = f'{cfg.tb_path}{experiment}/{creation_time}/'

    # @property
    # def base_path(self):
    #     return f'{cfg.report_path}{self.experiment}/{self.creation_time}/'

    # @property
    # def models_path(self):
    #     return f'{self.base_path}models/'

    # @property
    # def results_path(self):
    #     return f'{self.base_path}/results/'

    def __post_init__(self):
        self.creation_time = now_str('yyyymmdd_hhmmss')

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
