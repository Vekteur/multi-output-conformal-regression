import warnings
from pathlib import Path
import sys

from omegaconf import OmegaConf
from dask.distributed import Client
from moc.configs.config import get_config
from moc import utils
from moc.runner import run_all


def main():
    config = OmegaConf.from_cli(sys.argv)
    config = get_config(config)
    OmegaConf.resolve(config)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Pretty print config using Rich library
    if config.get('print_config'):
        utils.print_config(config, resolve=True)
    # Set parallelization
    if config.nb_workers == 1:
        manager = 'sequential'
    if config.manager == 'dask':
        Client(n_workers=config.nb_workers, threads_per_worker=1, memory_limit=None)
    # Run experiments
    return run_all(config, manager=config.manager)


if __name__ == '__main__':
    main()
