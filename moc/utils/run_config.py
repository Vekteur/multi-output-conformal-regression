from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


@dataclass
class RunConfig:
    config: DictConfig
    dataset_group: str
    dataset: str
    run_id: int = 0
    seed: int = -1
    hparams: dict = None
    metrics: dict = None

    def __post_init__(self):
        pass

    @property
    def dataset_group_config(self):
        return self.config.dataset_groups[self.dataset_group]

    @property
    def config_path(self):
        return Path(self.config.log_dir) / 'config.yaml'

    @property
    def dataset_path(self):
        return Path(self.config.log_dir) / self.dataset_group / self.dataset

    @property
    def run_path(self):
        path = self.dataset_path
        return path / self.hparams_str() / str(self.run_id)

    @property
    def storage_path(self):
        return self.run_path / 'run_config.pickle'

    @property
    def checkpoints_path(self):
        return self.run_path / 'checkpoints'

    def hparams_str(self, ignore_posthoc_grid=True):
        hparams = self.hparams
        if hparams is None:
            return 'None'
        if ignore_posthoc_grid:
            hparams = {key: value for key, value in hparams.items() if key != 'posthoc_grid'}
        return ','.join(f'{key}={value}' for key, value in hparams.items())

    def summary_str(self, bold=False):
        run_id = self.run_id
        if bold:
            run_id = f'\033[1m{self.run_id}\033[0m'
        summary_dict = {
            'dataset': self.dataset,
            'run': run_id,
            'hparams': self.hparams_str(ignore_posthoc_grid=False),
        }
        return ','.join(f'{key}:{value}' for key, value in summary_dict.items())

    def to_series(self):
        return pd.Series(self.__dict__)
