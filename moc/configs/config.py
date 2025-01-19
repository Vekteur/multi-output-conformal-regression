from omegaconf import OmegaConf

from .general import general_config


def get_config(config=None):
    if config is None:
        config = OmegaConf.create()
    config_builders = [
        general_config,
    ]
    for config_builder in config_builders:
        config = OmegaConf.merge(config_builder(config), config)
    return config
