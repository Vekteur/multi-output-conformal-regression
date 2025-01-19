import logging
import math
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.distributions import AffineTransform
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

log = logging.getLogger('moc')


# Source: https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        """
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        self.transformer = AffineTransform(loc=self.mean_, scale=self.scale_ + self.epsilon).inv
        return self

    def transform(self, values):
        return self.transformer(values)

    def inverse_transform(self, values):
        return self.transformer.inv(values)


class ScaledDataset(Dataset):
    def __init__(self, dataset, scaler_x, scaler_y):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def scale(self, v, scaler):
        shape = v.shape
        if len(shape) == 1:
            v = v[None, :]
        scaled = scaler.transform(v)
        return scaled.reshape(shape)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.scale(x, self.scaler_x)
        y = self.scale(y, self.scaler_y)
        return x, y


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        rc=None,
        num_workers=0,
        pin_memory=False,
        seed=0,
    ):
        super().__init__()

        # This line allows to access `__init__` arguments with `self.hparams` attribute
        self.save_hyperparameters(logger=False, ignore='rc')
        self.dataset_group = rc.dataset_group
        self.dataset = rc.dataset
        self.rc = rc
        self.train_val_calib_test_split_ratio = rc.config.train_val_calib_test_split_ratio
        self.load_datasets()

    @abstractmethod
    def get_data(self):
        pass

    def make_scaled_dataset(self, ds):
        return ScaledDataset(ds, self.scaler_x, self.scaler_y)

    def subsample(self, x, y, max_size):
        N = x.shape[0]
        rng = np.random.RandomState(self.hparams.seed)
        train_ratio = self.train_val_calib_test_split_ratio[0]
        sample_idx = rng.choice(N, min(N, math.ceil(max_size / train_ratio)), replace=False)
        return x[sample_idx], y[sample_idx]

    def load_datasets(self):
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        max_size = 20000
        if self.rc.config.fast:
            max_size = 1000
        x, y = self.subsample(x, y, max_size=max_size)
        tensor_data = TensorDataset(x, y)
        self.total_size = len(tensor_data)

        # Convert ratios to number of elements in the dataset
        splits_size = np.array(self.train_val_calib_test_split_ratio) * len(tensor_data)
        log.debug(f'Total size: {len(tensor_data)}')
        log.debug(f'Splits size before: {splits_size}')
        calib_index = 2
        to_remove_from_calib = max(0, splits_size[calib_index] - 2048)
        splits_size[calib_index] -= to_remove_from_calib
        # Don't add points in splits that should be empty (e.g., interleaving is often empty)
        mask = (splits_size != 0) & (np.arange(len(splits_size)) != calib_index)
        splits_size[mask] += to_remove_from_calib / mask.sum()
        splits_size = splits_size.astype(int)
        splits_size[-1] = len(tensor_data) - splits_size[:-1].sum()
        log.debug(f'Splits size after: {splits_size}')

        (self.data_train, self.data_val, self.data_calib, self.data_test,) = random_split(
            dataset=tensor_data,
            lengths=splits_size.tolist(),
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        x, y = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x)
        self.scaler_y = StandardScaler().fit(y)

        if self.rc.config.normalize:
            self.data_train = self.make_scaled_dataset(self.data_train)
            self.data_val = self.make_scaled_dataset(self.data_val)
            self.data_calib = self.make_scaled_dataset(self.data_calib)
            self.data_test = self.make_scaled_dataset(self.data_test)

        # Make the size of the inputs accessible to the models
        first_x, first_y = self.data_train[0]
        self.input_dim = first_x.shape[0]
        self.output_dim = first_y.shape[0]

    def get_dataloader(self, dataset, drop_last=False, shuffle=False, batch_size=None):
        if batch_size is None:
            batch_size = self.rc.config.default_batch_size
        batch_size = min(len(dataset), batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.data_train, drop_last=True, shuffle=True)
    
    def val_dataloader(self):
        return self.get_dataloader(self.data_val)
    
    def calib_dataloader(self):
        return self.get_dataloader(self.data_calib)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)
