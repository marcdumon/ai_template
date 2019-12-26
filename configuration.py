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
from pprint import pprint
from typing import List

import yaml


@dataclass
class Config:
    """Dataclass with all configuration parameters. These remain fixed from experiment to experiment"""
    config_file: str = './config.yml'
    recipe_file: str = './recipe.yml'

    @staticmethod
    def save_yaml():
        default_cfg = Config()
        with open(default_cfg.config_file, 'w') as f:
            yaml.dump(default_cfg.__dict__, f, default_flow_style=False)
        return default_cfg

    @classmethod
    def load_yaml(cls, file=None):
        """Load the config yaml and returns a Config dataclass"""
        if not file: file = cls.config_file
        try:
            with open(file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**config)
        except FileNotFoundError:
            print("Config file doesn't exist. Run Config.create_empty_yaml to create cls.config_file with default settings")
            return Config()


cfg = Config.load_yaml()
Config.save_yaml()


@dataclass()
class Recipe:
    """
    A dataclass with all the parameters that might vary from one experiment to the other or from one stage of an experiment
    to the other stage
    """
    seed: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    dp: list = field(default_factory=list)

    @staticmethod
    def create_empty_yaml():
        default_rcp = Recipe()
        with open(cfg.recipe_file, 'w') as f:
            yaml.dump(default_rcp.__dict__, f, default_flow_style=False)
        return default_rcp

    @classmethod
    def load_yaml(cls, file=None):
        """Load the recipe yaml and returns a Recipe dataclass"""
        if not file: file = cfg.recipe_file
        try:
            with open(file, 'r') as f:
                recipe = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**recipe)
        except FileNotFoundError:
            print("Recipe file doesn't exist. Run Recipe.create_empty_yaml to create cfg.recipe_file with default settings")
            print("Returning default recipe now")
            return Recipe()


rcp = Recipe.load_yaml()

if __name__ == '__main__':
    # x = Recipe()
    x = cfg
    pprint(x)
    x = rcp
    pprint(x)

    # Recipe.create_empty_yaml()
