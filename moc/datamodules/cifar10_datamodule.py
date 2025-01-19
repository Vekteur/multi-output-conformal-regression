import numpy as np
import torch
from torchvision import datasets, transforms

from .base_datamodule import BaseDataModule


class CIFAR10DataModule(BaseDataModule):
    def get_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_data = datasets.CIFAR10('data/cifar10/', train=False, download=True, transform=transform)
        # For compatability with other functions, we treat the image as unidimensional and transform the image
        # to the correct shape inside image models.
        x = np.array(test_data.targets).reshape(-1, 1)
        y = test_data.data
        assert y.shape[1:] == (32, 32, 3)
        y = np.moveaxis(y, -1, 1)
        assert y.shape[1:] == (3, 32, 32) # (C, H, W)
        y = np.reshape(y, (y.shape[0], -1))
        # We truncate the dataset since we don't need a training dataset for the pretrained model
        size = 5000
        x = x[:size]
        y = y[:size]
        y = y.astype(np.float32) / 255.
        return x, y
