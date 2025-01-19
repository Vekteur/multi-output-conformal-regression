"""
Code adapted from https://github.com/LeoGrin/tabular-benchmark/blob/main/data/data_utils.py
"""

import pickle
import logging

import pandas as pd
import numpy as np

log = logging.getLogger('moc')


class InvalidDataset(Exception):
    pass


def remove_high_cardinality(x, y, categorical_mask, threshold=20):
    """Remove categorical columns with a high number of categories"""
    high_cardinality_mask = x.nunique().to_numpy() > threshold
    to_remove = categorical_mask & high_cardinality_mask
    n_removed = np.sum(to_remove)
    x = x.drop(x.columns[to_remove], axis=1)
    categorical_mask = categorical_mask[~to_remove]
    return x, y, categorical_mask, n_removed


def remove_pseudo_categorical(x, y, categorical_mask, min_unique=10):
    """Remove numerical columns where most values are the same"""
    #num_cols = set(x.select_dtypes(include='number').columns)
    #num_mask = x.columns.isin(num_cols)
    num_mask = ~categorical_mask
    pseudo_categorical_cols_mask = x.nunique() < min_unique
    to_remove = num_mask & pseudo_categorical_cols_mask
    n_removed = np.sum(to_remove)
    x = x.drop(x.columns[to_remove], axis=1)
    return x, y, n_removed


def remove_missing_values(x, y, threshold=0.2):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(x).mean(axis=0) > threshold
    x = x.drop(x.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(x).any(axis=1)
    x = x[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return x, y, np.sum(missing_cols_mask), np.sum(missing_rows_mask)


def save_dataset(x, y, path):
    with open(path / 'x.npy', 'wb') as f:
        pickle.dump(x, f)
    with open(path / 'y.npy', 'wb') as f:
        pickle.dump(y, f)


def load_dataset(path):
    with open(path / 'x.npy', 'rb') as f:
        x = pickle.load(f)
    with open(path / 'y.npy', 'rb') as f:
        y = pickle.load(f)
    return x, y


def identify_categorical_columns(x, mask=None, threshold=10):
    if mask is None:
        mask = pd.Series(False, index=x.columns)
    mask = (x.dtypes == 'object') | (x.dtypes == 'bool')
    nunique = x.nunique()
    mask |= (x.dtypes == 'int64') & (nunique <= threshold)
    mask |= (nunique == 2)
    return mask


def preprocess(x, y, categorical_mask=None):
    # For some datasets, y is a ndarray instead of a dataframe
    if type(y) == np.ndarray:
        y = pd.DataFrame(y)
    n_columns = len(x.columns)
    categorical_mask = identify_categorical_columns(x, categorical_mask)
    x, y, categorical_mask, n_high_cardinality = remove_high_cardinality(x, y, categorical_mask)
    x, y, n_pseudo_categorical = remove_pseudo_categorical(x, y, categorical_mask)
    x, y, n_missing_cols, n_missing_rows = remove_missing_values(x, y)
    n_removed = n_high_cardinality + n_pseudo_categorical + n_missing_cols
    log.debug(f"Removed columns: {n_high_cardinality} + {n_pseudo_categorical} + {n_missing_cols} = {n_removed} over {n_columns}")
    log.debug(f"Removed rows: {n_missing_rows}")
    if x.columns.empty:
        raise InvalidDataset('No remaining columns')
    x = pd.get_dummies(x)
    if len(x) < 100:
         raise InvalidDataset('Not enough rows in the dataset')
    x, y = x.to_numpy('float32'), y.to_numpy('float32')
    assert np.isnan(x).sum() == 0 and np.isnan(y).sum() == 0
    assert np.isinf(x).sum() == 0 and np.isinf(y).sum() == 0
    return x, y, categorical_mask
