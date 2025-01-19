from datetime import datetime
from pathlib import Path
from enum import IntEnum

from omegaconf import OmegaConf


class PrecomputationLevel(IntEnum):
    NONE = 0
    MODELS = 1
    RESULTS = 2


def get_log_dir(config):
    assert config.name not in ['fast', 'debug'], 'config.name cannot be "fast" or "debug"'
    optional_dirs = []
    for dir in ['fast', 'debug']:
        if config.get(dir):
            optional_dirs.append(dir)
    log_dir = Path(config.log_base_dir)
    if optional_dirs:
        log_dir /= '-'.join(optional_dirs)
    if config.name is not None:
        log_dir /= config.name
    else:
        log_dir /= datetime.now().strftime(r'%Y-%m-%d')
        log_dir /= datetime.now().strftime(r'%H-%M-%S')
    return log_dir


def general_config(config):
    work_dir = Path('.').resolve()
    default_config = OmegaConf.create(
        dict(
            work_dir=str(work_dir),
            data_dir=str(work_dir / 'data'),
            log_base_dir=str(work_dir / 'logs'),
            # Name of the experiment
            # If no name is specified, the name will be the current date and time
            name=None,
            device='cpu',
            alpha=0.2,
            train_val_calib_test_split_ratio=(0.4, 0.1, 0.3, 0.2),
            default_batch_size=256,
            tuning_type='default',
            print_config=True,
            progress_bar=False,
            # Use cache for faster metrics computation and better comparison between conformalizers
            use_cache=True,
            # Whether to noramlize the data to have mean 0 and std 1
            normalize=True,
            # Whether to remove checkpoints to avoid using a large amount of disk space
            remove_checkpoints=False,
            # Indicates which subset of datasets to select
            datasets='default',
            # If True, the experiment will be repeated only once on a few batches of the dataset
            fast=False,
            debug=False,
            # Which manager to use for parallelization in ['dask', 'joblib', 'sequential']
            # 'sequential' is always selected when nb_workers=1
            manager='joblib', 
            nb_workers=1,
            # This selects runs with run_id in the range [start_repeat_tuning, repeat_tuning)
            start_repeat_tuning=0,
            repeat_tuning=10,
            precomputation_level=PrecomputationLevel.RESULTS,
        )
    )
    config = OmegaConf.merge(default_config, config)
    if config.fast:
        config.repeat_tuning = 1
    log_dir = get_log_dir(config)
    if config.name == 'unnamed' and log_dir.exists():
        raise RuntimeError('Unnamed experiment already exists')
    config.log_dir = str(log_dir)
    return config
