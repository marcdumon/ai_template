# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - configuration.py
# md
# --------------------------------------------------------------------------------------------------------
"""
Manages the configuration of the environemet and the recipe for running the machine.
It reads a yaml files into dataclasses.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from pprint import pprint
from time import sleep
from typing import List

import yaml

from my_tools.tools import now_str

'''
Why using dataclass iso dictionary ?
    - structure:
        module configuration.py
            DC()
            cd = DC()
        module main_app.py
            from configuration import dc
            dc.lr = 10                  dic['lr'] = 10
                                        run(dic)
        module machine.py
            from configuration import dc
                lr = dc.lr              lr = dic['lr']
    - getting and setting dataclass iso dictionary: 
        - lr = dc.lr iso lr = dic['ld']
        - easy saving dataclass
    - using of gridsearch: 
        for lr in [1e-2, 1e-3, ...]:
            for drop in [.1, .2, ...]:
                dc.lr = lr                dic['lr] = lr
                dc.drop = drop            dic['drop'] = drop
                run()                     run(dic)
    - using stages:
        dc1 = DC()
        dc2 = DC()
        for dc in [dc1, dc2]:
            run()
    - save and load:
        dc.save()
        dc.load()
'''


@dataclass
class Config:
    """Dataclass with all configuration parameters. These remain fixed from experiment to experiment"""
    default_config_file: str = './default_config.yml'
    default_recipe_file: str = './default_recipe.yml'

    # checkpoint_path: str = '/media/md/Development/'
    checkpoint_path: str = './'

    @staticmethod
    def save_default_yaml():
        default_rcp = Config().save_yaml(Config.default_config_file)

    def save_yaml(self, file=f'{checkpoint_path}config.yml'):
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=f'{checkpoint_path}config.yml'):
        """Load the recipe yaml and returns a Recipe dataclass"""
        try:
            with open(file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**config)
        except FileNotFoundError:
            print("Recipe file doesn't exist.")
            print("Returning default config instead")
            return Config()


cfg = Config()


@dataclass()
class Recipe:  # Prescription, Ingredient, ModusOperandi
    """
    A dataclass with all the parameters that might vary from one experiment to the other or from one stage of an experiment
    to the other stage
    """
    experiment: str = 'exp'
    stage: int = 1

    seed: int = 19640601
    lr: float = 3e-3
    dropout: float = 0.5
    creation_time: str = now_str('yyyymmdd_hhmmss')

    @staticmethod
    def save_default_yaml():
        default_rcp = Recipe().save_yaml(cfg.default_recipe_file)

    def save_yaml(self, file=f'{cfg.checkpoint_path}rcp_{experiment}_{stage}.yml'):
        with open(file, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        return self

    @classmethod
    def load_yaml(cls, file=f'{cfg.checkpoint_path}rcp_{experiment}_{stage}.yml'):
        """Load the recipe yaml and returns a Recipe dataclass"""
        try:
            with open(file, 'r') as f:
                recipe = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**recipe)
        except FileNotFoundError:
            print("Recipe file doesn't exist.")
            print("Returning default recipe instead")
            return Recipe()


rcp = Recipe()

if __name__ == '__main__':
    print(cfg)
    # sleep(5)
    cfg.save_yaml()

    cfg.load_yaml()
    print(cfg)
