import pickle
from collections import defaultdict
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from pandas.io.formats.style_render import _escape_latex
import scipy.stats
import yaml
from omegaconf import OmegaConf
import torch

from moc.configs.config import get_config
from moc.configs.datasets import get_dataset_groups
from moc.datamodules import load_datamodule
from moc.utils.general import filter_dict
from moc.utils.run_config import RunConfig
from moc.conformal.conformalizers_manager import conformalizers
from moc.analysis.helpers import create_name_from_dict, main_metrics, other_metrics


# Small hack that allows to pickle the logs even if some pickled class does not exist anymore.
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            return object


def load_config(path):
    with open(Path(path) / 'config.yaml', 'r') as f:
        return OmegaConf.create(yaml.load(f, Loader=yaml.Loader), flags={'allow_objects': True})


def make_df(config, dataset_group, dataset, reload=True):
    dataset_path = Path(config.log_dir) / dataset_group / dataset
    if not dataset_path.exists():
        return None
    df_path = dataset_path / 'df.pickle'
    if not reload and df_path.exists():
        with open(df_path, 'rb') as f:
            return pickle.load(f)
    series_list = []
    for run_config_path in dataset_path.rglob('run_config.pickle'):
        with open(run_config_path, 'rb') as f:
            rcs = CustomUnpickler(f).load()
        for rc in rcs:
            series_list.append(rc.to_series())
    if len(series_list) == 0:
        return None
    df = pd.concat(series_list, axis=1).T
    with open(df_path, 'wb') as f:
        pickle.dump(df, f)
    return df


def load_df(config, dataset_group=None, dataset=None, reload=True):
    assert config is not None
    dfs = []
    path = Path(config.log_dir)
    if dataset_group is None:
        dataset_groups = [p for p in path.iterdir() if p.is_dir()]
    else:
        dataset_groups = [dataset_group]
    for curr_dataset_group in dataset_groups:
        path = Path(config.log_dir) / curr_dataset_group
        if dataset is None:
            datasets = [p for p in path.iterdir() if p.is_dir()]
        else:
            datasets = [dataset]
        for curr_dataset in datasets:
            df = make_df(config, curr_dataset_group, curr_dataset, reload=reload)
            if df is not None:
                dfs.append(df)
    if not dfs:
        raise RuntimeError('Dataframe not found')
    df = pd.concat(dfs)
    return df


def add_total_time_metric(df):
    names = df.index.names
    df_temp = df.reset_index()
    df1 = df_temp.query('metric == "score_time"').assign(metric='total_time').set_index(names)
    df2 = df_temp.query('metric == "test_coverage_time"').assign(metric='total_time').set_index(names)
    df_total_time = df1 + df2
    df = pd.concat([df, df_total_time], axis=0)
    return df


def get_metric_df(config, df):
    hparams = [
        'model', 
        'posthoc_method', 
        'mixture_size', 
        'posthoc_correction_factor', 
        'posthoc_n_samples', 
        'posthoc_n_samples_mc', 
        'posthoc_n_samples_ref'
    ]
    for hparam in hparams:
        df[hparam] = df.apply(lambda df: df.hparams.get(hparam, None), axis=1)
    metrics = main_metrics + other_metrics
    for metric in metrics:
        df[metric] = df.apply(lambda df: df.metrics.get(metric, np.nan), axis=1)
    df = df.drop(columns=['hparams', 'metrics', 'config'])

    df_ds = get_datasets_df(config, reload=False)
    order = df_ds.sort_values('Nb instances').reset_index()['Dataset']
    df['dataset'] = pd.Categorical(df['dataset'], order)
    df = df.sort_values('dataset')

    other_columns = [col for col in df.columns if col not in metrics]
    df = df.set_index(other_columns)
    df = df.stack(dropna=False).rename_axis(index={None: 'metric'}).to_frame(name='value')
    names = df.index.names
    df = df.reset_index()
    df['metric'] = pd.Categorical(df['metric'], metrics)
    df['posthoc_method'] = pd.Categorical(df['posthoc_method'], conformalizers.keys())
    df.loc[df['metric'].isin(['cond_cov_x_error', 'cond_cov_z_error']), 'value'] *= 100
    df.loc[df['model'] == 'Mixture', 'model'] = df['model'] + '-' + df['mixture_size'].astype(pd.Int32Dtype()).astype(str)
    
    model_name_partial = partial(create_name_from_dict, config=config)
    df['name'] = df.apply(model_name_partial, axis='columns').astype('string')

    df = df.set_index(names + ['name'])
    df = add_total_time_metric(df)
    df[df.eval('metric == "wsc" and value.isna()')] = 1
    return df


class Highlighter:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def order_metric(self, data, metric):
        def get_mean(x): # Get the mean
            if pd.isna(x):
                return np.nan
            mean, _ = x
            return mean
        data = data.map(get_mean)
        if metric in ['coverage', 'wsc']:
            return (data - (1 - self.alpha)).abs()
        return data

    def highlight_min_per_metric(self, data):
        """
        Highlight the minimum value for each metric, supposing that the index is 'dataset' and columns are ('metric', 'posthoc_method').
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for metric in data.columns.get_level_values('metric').unique():
            metric_data = data[metric]
            min_values = self.order_metric(metric_data, metric).min(axis=1)
            for col in metric_data.columns:
                is_min = self.order_metric(metric_data[col], metric) == min_values
                styles.loc[is_min, (metric, col)] = 'font-weight: bold;'
        return styles

    def highlight_min_per_dataset(self, data):
        """
        Highlight the minimum value for each metric, supposing that the index is ('dataset', 'posthoc_method') and columns are 'metric'.
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for dataset in data.index.get_level_values('dataset').unique():
            dataset_data = data.loc[dataset]
            for metric in data.columns.get_level_values('metric').unique():
                metric_data = dataset_data[metric]
                min_values = self.order_metric(metric_data, metric).min()
                is_min = self.order_metric(metric_data, metric) == min_values
                styles.loc[dataset, (metric)] = np.where(is_min, 'font-weight: bold;', '')
        return styles


def agg_mean_sem(x):
    mean = np.mean(x)
    std = None
    if len(x) > 1:
        std = scipy.stats.sem(x, ddof=1)
    return (mean, std)


def format_cell_latex(x):
    if pd.isna(x):
        return 'NA'
    mean, sem = x
    if pd.isna(mean):
        return 'NA'
    if np.isposinf(mean):
        return r'$\infty$'
    if np.isneginf(mean):
        return r'$-\infty$'
    s = rf'\text{{{mean:#.3}}}'
    if sem is not None:
        sem = float(sem)
        s += rf'_{{\text{{{sem:#.2}}}}}'
    s = f'${s}$'
    return s


def format_cell_jupyter(x, add_sem=False):
    if pd.isna(x):
        return 'NA'
    mean, sem = x
    if pd.isna(mean):
        return 'NA'
    elif np.isposinf(mean):
        return '∞'
    elif np.isneginf(mean):
        return '-∞'
    else:
        s = f'{mean:#.3}'
    if add_sem and sem is not None:
        s += f' ± {sem:#.2}'
    return s


def compute_datasets_df(config):
    data = defaultdict(list)
    for dataset_group, datasets in get_dataset_groups(config.datasets).items():
        for dataset in datasets:
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
            )
            datamodule = load_datamodule(rc)
            data_train = datamodule.data_train
            nb_instances = datamodule.total_size
            first_item = next(iter(data_train))
            x, y = first_item
            x_dim, y_dim = x.shape[0], y.shape[0]
            description = {
                'Group': dataset_group,
                'Dataset': dataset,
                'Nb instances': nb_instances,
                'Nb features': x_dim,
                'Nb targets': y_dim,
            }
            for key, value in description.items():
                data[key].append(value)
    return pd.DataFrame(data).set_index(['Group', 'Dataset'])


def get_datasets_df(config, reload=False):
    path = Path(config.log_dir) / 'datasets_df.pickle'
    if not reload and path.exists():
        return pd.read_pickle(path)
    df = compute_datasets_df(config)
    df.to_pickle(path)
    return df


def get_value_counts(y):
    values, counts = torch.unique(y, return_counts=True, dim=0)
    indices = counts.argsort(descending=True)
    values, counts = values[indices], counts[indices]
    return values, counts

def get_info(y):
    values, counts = get_value_counts(y)
    N = y.shape[0]
    proportions = counts / N
    d = {
        'Proportion of top 1 classes': proportions[:1].sum().item(),
        'Proportion of top 10 classes': proportions[:10].sum().item(),
        'Proportion of duplicated values': proportions[counts > 1].sum().item(),
    }
    return d

def apply_duplication_style(df):
    def bold_values(val):
        return 'font-weight: bold' if val > 0.5 else ''
    cols = [
        'Proportion of top 1 classes', 
        'Proportion of top 10 classes', 
        'Proportion of duplicated values'
    ]
    return df.style.map(bold_values, subset=cols).format(precision=3)

def compute_duplication_df(config):
    data = defaultdict(list)
    for dataset_group, datasets in get_dataset_groups(config.datasets).items():
        for dataset in datasets:
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
            )
            datamodule = load_datamodule(rc)
            x, y = datamodule.data_train[:]
            data['Group'].append(dataset_group)
            data['Dataset'].append(dataset)
            data['Nb instances'].append(datamodule.total_size)
            data['Nb features'].append(x.shape[1])
            data['Nb targets'].append(y.shape[1])
            for key, value in get_info(y).items():
                data[key].append(value)
    df = pd.DataFrame(data).set_index(['Group', 'Dataset'])
    return df

def get_duplication_df(config, reload=False):
    path = Path(config.log_dir) / 'duplication_df.pickle'
    if not reload and path.exists():
        return pd.read_pickle(path)
    df = compute_duplication_df(config)
    df.to_pickle(path)
    return df


def latex_style(styler):
    if styler.columns.names != [None]:
        styler.columns.names = list(map(_escape_latex, styler.columns.names))
    return (styler
            .format_index(escape='latex', axis=0)
            .format_index(escape='latex', axis=1)
    )


def to_latex(styler, path=None, hrules=True, multicol_align='c', multirow_align='t', convert_css=True, **kwargs):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    return latex_style(styler).to_latex(
        path, hrules=hrules, multicol_align=multicol_align, multirow_align=multirow_align, convert_css=convert_css, **kwargs
    )
