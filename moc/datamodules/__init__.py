from .real_datamodule import RealDataModule
from .toy_datamodule import ToyDataModule
from .cifar10_datamodule import CIFAR10DataModule
from moc.configs.datasets import toy_dataset_groups, real_dataset_groups


def get_datamodule(group):
    if group in toy_dataset_groups:
        return ToyDataModule
    elif group in real_dataset_groups:
        return RealDataModule
    elif group == 'cifar10':
        return CIFAR10DataModule
    raise ValueError(f'Unknown datamodule {group}')


def load_datamodule(rc):
    datamodule_cls = get_datamodule(rc.dataset_group)
    return datamodule_cls(
        rc=rc,
        seed=2000 + rc.run_id,
    )
