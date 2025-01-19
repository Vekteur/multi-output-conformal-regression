from copy import copy
import logging

from moc.utils.general import elapsed_timer
from . import conformalizers

log = logging.getLogger('moc')


class ConformalizerModule:
    def __init__(self, datamodule, model, rc, hparams, cache_calib):
        log.info(f'Initializing {hparams}')
        self.hparams = hparams
        dl = datamodule.calib_dataloader()
        kwargs = hparams.copy()
        method = kwargs.pop('method')
        with elapsed_timer() as timer:
            self.conformalizer = conformalizers[method](dl, model, cache_calib=cache_calib, rc=rc, **kwargs)
        self.metrics = {
            'score_time': timer() + cache_calib.get_time(self.conformalizer.used_cache_keys()),
        }

    def make_run_config(self, rc):
        rc = copy(rc)
        hparams_with_prefix = {f'posthoc_{key}': value for key, value in self.hparams.items()}
        rc.hparams = {**rc.hparams, **hparams_with_prefix}
        # Convert any defaultdict to dict due to a bug (https://github.com/python/cpython/issues/79721)
        rc.metrics = dict(self.metrics)
        return rc


class ConformalizersManager:
    def __init__(self, datamodule, model, rc, posthoc_grid, cache_calib=None):
        self.modules = [ConformalizerModule(datamodule, model, rc, hparams, cache_calib) for hparams in posthoc_grid]

    def get_module(self, hparams):
        for module in self.modules:
            if module.hparams == hparams:
                return module
        raise ValueError(f'No module with hparams {hparams} found.')

    def make_run_configs(self, rc):
        return [module.make_run_config(rc) for module in self.modules]
