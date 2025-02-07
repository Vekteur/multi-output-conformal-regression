import logging
import pickle
import shutil
import traceback
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import as_completed, get_client
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities.warnings import PossibleUserWarning

from moc.models.train import run
from moc.models.tuning import get_tuning
from moc.utils.run_config import RunConfig
from moc.configs.datasets import get_dataset_groups
from moc.configs.general import PrecomputationLevel

log = logging.getLogger('moc')


def mute_cumbersome_logging():
    logging.getLogger('lightning_lite.utilities.seed').setLevel(logging.WARNING)
    logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)
    # logging.getLogger('torch.distributed.nn.jit.instantiator').setLevel(logging.WARNING)
    logging.getLogger('distributed.diskutils').setLevel(logging.WARN)
    warnings.filterwarnings('ignore', '.*Unmanaged memory use is high.*')
    warnings.filterwarnings(
        'ignore',
        r'.*The `srun` command is available on your system but is not used.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        'ignore',
        r'.*GPU available but not used\. Set `accelerator` and `devices` using.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        'ignore',
        r'.*does not have many workers which may be a bottleneck\. Consider increasing the value of the.*',
        category=PossibleUserWarning,
    )


def train_and_save(rc, hparams, pm):
    logging.basicConfig(level=logging.INFO)
    mute_cumbersome_logging()

    rc.hparams = hparams
    if rc.config.precomputation_level >= PrecomputationLevel.RESULTS and rc.storage_path.exists():
        # # Workaround to overwrite results with specific hyperparameters
        # if rc.hparams['model'] == 'DRF-KDE':
        #     pass # We overwrite the results
        # else:
        with open(rc.storage_path, 'rb') as f:
            return pickle.load(f)

    index = 0 if pm is None else pm.request().result()
    # Run training
    rc, rcs = run(rc, index)

    if pm is not None:
        pm.free(index).result()

    rc.storage_path.parent.mkdir(parents=True, exist_ok=True)
    if rc.config.remove_checkpoints:
        assert len(list(rc.checkpoints_path.rglob('*'))) <= 2, list(rc.checkpoints_path.rglob('*'))
        # In case of error, check that I am not running the same runs (with same hyperparameters) concurrently!
        shutil.rmtree(rc.checkpoints_path)
    with open(rc.storage_path, 'wb') as f:
        # Don't save the whole config. This should save a lot of space.
        # However, it should be readded after loading the RunConfig again.
        for rc_posthoc in rcs:
            rc_posthoc.config = None
        pickle.dump(rcs, f)
    return rc


class PositionManager:
    """
    The Position Manager allows to keep track of which process is currently running.
    It can be useful e.g. to show multiple progress bars in parallel.
    """
    def __init__(self, size):
        self.slots = [False for _ in range(size)]

    def free(self, i):
        self.slots[i] = False

    def request(self):
        for i, slot in enumerate(self.slots):
            if not slot:
                self.slots[i] = True
                return i
        log.info('No slot available')
        return 0


class Runner:
    def __init__(self, config, manager='sequential'):
        self.config = config
        assert manager in ['sequential', 'dask', 'joblib']
        self.manager = manager
        self.tasks = []
        if self.manager == 'dask':
            pm_future = self.submit(
                PositionManager,
                self.config.nb_workers,
                actor=True,
            )
            self.pm = pm_future.result()
        else:
            self.pm = None
    
    def submit(self, fn, *args, priority=None, **kwargs):
        if self.manager == 'dask':
            return get_client().submit(fn, *args, **kwargs, priority=priority)
        elif self.manager == 'joblib':
            return (fn, args, kwargs)
        else:
            return fn(*args, **kwargs)

    def train_in_parallel(self, rc, hparams, priority):
        return self.submit(
            train_and_save,
            rc,
            hparams,
            self.pm,
            priority=priority,
        )

    def run_grid_search(self, rc, priority):
        grid = get_tuning(rc.config)
        for hparams in grid:
            future_rc = self.train_in_parallel(rc, hparams, priority)
            self.tasks.append(future_rc)

    def close(self):
        if self.manager == 'dask':
            for future in as_completed(self.tasks):
                if future.status == 'error':
                    message = (
                        "Error in parallel task\n"
                        f"{'=' * 60}\n"
                        "Traceback\n"
                        f"{'=' * 60}\n"
                        f"{future.traceback()}\n"
                        f"Exception: {future.exception()}"
                    )
                    log.info(message)
        elif self.manager == 'joblib':
            Parallel(n_jobs=self.config.nb_workers)(delayed(wrapped_fn)(fn, *args, **kwargs) for fn, args, kwargs in self.tasks)


def wrapped_fn(fn, *args, **kwargs):
    # This function is used to catch exceptions in parallel tasks without stopping the other tasks
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print('=== Start of error in wrapped function ===', flush=True)
        print('args:', flush=True)
        for arg in args:
            if type(arg) == RunConfig:
                print(arg.summary_str(bold=False))
            else:
                print(arg)
        print('kwargs:', flush=True)
        print(kwargs)
        print('Traceback:', flush=True)
        traceback.print_exc()
        print('=== End of error in wrapped function ===', flush=True)
        # We do not raise the exception here to avoid stopping the other tasks
        #raise e


def run_all(config: DictConfig, manager='sequential'):
    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)
    OmegaConf.save(config, Path(config.log_dir) / 'config.yaml')

    runner = Runner(config, manager=manager)
    priority = 0
    for run_id in range(config.start_repeat_tuning, config.repeat_tuning):
        for dataset_group, datasets in get_dataset_groups(config.datasets).items():
            # log.info(f"Dataset group: \033[1;4m{dataset_group}\033[0m")
            for dataset in datasets:
                # log.info(f"  Dataset: \033[4m{dataset}\033[0m")
                rc = RunConfig(
                    config=config,
                    dataset_group=dataset_group,
                    dataset=dataset,
                    run_id=run_id,
                )
                runner.run_grid_search(rc, priority)
                priority -= 1
    runner.close()
