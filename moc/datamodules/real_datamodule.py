from pathlib import Path

import pandas as pd
import numpy as np
from scipy.io import arff

from .base_datamodule import BaseDataModule
from .preprocessing import preprocess


def get_data_camehl(data_dir, name):
    targets_dict = {
        'households': ['inc', 'food', 'house', 'utili'],
    }

    data_path = Path(data_dir)
    path = data_path / 'camehl' / f'{name}.csv'
    df = pd.read_csv(path, index_col=0)
    df = df.drop(columns=['newid', 'inc.a'])
    targets = targets_dict[name]
    x = df[df.columns.difference(targets)]
    y = df[targets]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


def get_data_cevid(data_dir, name):
    nb_targets_dict = {
        'air': 6,
        'births1': 2,
        'births2': 4,
        'wage': 2,
    }

    data_path = Path(data_dir)
    path = data_path / 'cevid' / f'{name}.csv'
    df = pd.read_csv(path)
    nb_targets = nb_targets_dict[name]
    x = df.iloc[:, :-nb_targets]
    y = df.iloc[:, -nb_targets:]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


def get_data_del_barrio(data_dir, name):
    nb_targets_dict = {
        'ansur2': 2,
        'calcofi': 2,
    }

    data_path = Path(data_dir)
    path = data_path / 'del_barrio' / f'{name}.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    nb_targets = nb_targets_dict[name]
    x = df.iloc[:, :-nb_targets]
    y = df.iloc[:, -nb_targets:]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


def get_data_feldman(data_dir, name):
    targets_dict = {
        'bio': ['F7', 'F9'],
        'blog_data': [60, 280],
        'house': ['lat', 'price'],
        'meps_19': ['K6SUM42', 'UTILIZATION_reg'],
        'meps_20': ['K6SUM42', 'UTILIZATION_reg'],
        'meps_21': ['K6SUM42', 'UTILIZATION_reg'],
    }

    data_path = Path(data_dir)
    path = data_path / 'feldman' / f'{name}.csv'
    if name == 'blog_data':
        df = pd.read_csv(path, header=None)
    elif name in ['bio', 'house']:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, index_col=0)
    targets = targets_dict[name]
    x = df[df.columns.difference(targets)]
    y = df[targets]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


def get_data_mulan(data_dir, name):
    nb_targets_dict = {
        "atp1d": 6,
        "atp7d": 6,
        "oes97": 16,
        "oes10": 16,
        "rf1": 8,
        "rf2": 8,
        "scm1d": 16,
        "scm20d": 16,
        "edm": 2,
        "sf1": 3,
        "sf2": 3,
        "jura": 3,
        "wq": 14,
        "enb": 2,
        "slump": 3,
        "andro": 6,
        "osales": 12,
        "scpf": 3
    }

    data_path = Path(data_dir)
    path = data_path / 'mulan' / f'{name}.arff'
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    nb_targets = nb_targets_dict[name]
    x = df.iloc[:, :-nb_targets]
    y = df.iloc[:, -nb_targets:]
    x_names = meta.names()[:-nb_targets]
    categorical_mask = np.array([meta[name][0] == 'nominal' for name in x_names])
    x, y, categorical_mask = preprocess(x, y, categorical_mask)
    return x, y


def get_data_wang(data_dir, name):
    nb_targets_dict = {
        'energy': 2,
        'taxi': 2,
    }

    data_path = Path(data_dir)
    path = data_path / 'wang' / f'{name}.csv'
    df = pd.read_csv(path)
    nb_targets = nb_targets_dict[name]
    x = df.iloc[:, :-nb_targets]
    y = df.iloc[:, -nb_targets:]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


def get_data(data_dir, group, name):
    if group == 'camehl':
        return get_data_camehl(data_dir, name)
    elif group == 'cevid':
        return get_data_cevid(data_dir, name)
    elif group == 'del_barrio':
        return get_data_del_barrio(data_dir, name)
    elif group == 'feldman':
        return get_data_feldman(data_dir, name)
    elif group == 'mulan':
        return get_data_mulan(data_dir, name)
    elif group == 'wang':
        return get_data_wang(data_dir, name)
    else:
        raise ValueError(f'Unknown group: {group}')


class RealDataModule(BaseDataModule):
    def get_data(self):
        return get_data(self.rc.config.data_dir, self.dataset_group, self.dataset)
