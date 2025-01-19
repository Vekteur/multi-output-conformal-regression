import logging

import torch

from moc.datamodules import get_datamodule
from moc.conformal.conformalizers_manager import ConformalizersManager
from moc.metrics.metrics_computer import MetricsComputer
from moc.utils.general import seed_everything
from moc.configs.general import PrecomputationLevel
from lightning import Trainer
from . import models, trainers

log = logging.getLogger('moc')


def train(rc, datamodule):
    model_path = rc.checkpoints_path / 'best.pth'

    model_kwargs = rc.hparams.copy()
    model_name = model_kwargs.pop('model')
    model_cls = models[model_name]
    model_kwargs['input_dim'] = datamodule.input_dim
    model_kwargs['output_dim'] = datamodule.output_dim
    if model_cls.output_type() == 'quantile':
        model_kwargs['alpha'] = torch.tensor([rc.config.alpha / 2, 1 - rc.config.alpha / 2])
    model = model_cls(**model_kwargs)
    
    trainer_cls = trainers[model_name]
    load_cpkt_condition = (
        rc.config.precomputation_level >= PrecomputationLevel.MODELS 
        and model_path.exists()
        and trainer_cls == Trainer
    )
    if load_cpkt_condition:
        model = model_cls.load_from_checkpoint(model_path)
        log.info(f'Finished loading {rc.summary_str()}')
    else:
        trainer = trainers[model_name](rc=rc)
        trainer.fit(model=model, datamodule=datamodule)
        rc.checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
        if trainer_cls == Trainer:
            trainer.save_checkpoint(model_path)
        log.info(f'Finished training {rc.summary_str()}')
    return model


def run(rc, process_index):
    log.info(f'Starting {rc.summary_str()}')
    datamodule_cls = get_datamodule(rc.dataset_group)
    datamodule = datamodule_cls(
        rc=rc, 
        seed=2000 + rc.run_id, 
    )

    seed_everything(1000 + rc.run_id)
    posthoc_grid = rc.hparams.pop('posthoc_grid')

    model = train(rc, datamodule)
    # The device of the model can change to CPU after training, so we transfer to the correct device
    model.to(rc.config.device)

    metrics_computer = MetricsComputer(model, rc.config.alpha, datamodule, use_cache=rc.config.use_cache)

    conformalizers_manager = ConformalizersManager(
        datamodule=datamodule, 
        model=model, 
        rc=rc,
        posthoc_grid=posthoc_grid,
        cache_calib=metrics_computer.cache_calib,
    )

    for posthoc_module in conformalizers_manager.modules:
        log.info(f'Computing metrics for {posthoc_module.hparams}')
        metrics = metrics_computer.compute_metrics(posthoc_module.conformalizer)
        posthoc_module.metrics.update(metrics)

    rcs = conformalizers_manager.make_run_configs(rc)

    log.info(f'Finished {rc.summary_str()}')
    return rc, rcs
