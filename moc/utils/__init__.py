import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import lightning.pytorch as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        'debug',
        'info',
        'warning',
        'error',
        'exception',
        'fatal',
        'critical',
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Optional[Sequence[str]] = None,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = 'dim'
    tree = rich.tree.Tree('config', style=style, guide_style=style)

    if fields is None:
        fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    rich.print(tree)

    path = Path(config.log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'config_tree.log', 'w') as fp:
        rich.print(tree, file=fp)
