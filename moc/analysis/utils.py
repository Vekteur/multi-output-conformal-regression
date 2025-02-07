from moc.models import models, OracleModel
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.models.trainers.default_trainer import DefaultTrainer


def train_lightning(rc, model_name, datamodule, ckpt_suffix=None, retrain=False, **kwargs):
    if ckpt_suffix is None:
        ckpt_name = model_name
    else:
        ckpt_name = f'{model_name}_{ckpt_suffix}'
    ckpt_path = rc.checkpoints_path / f'{ckpt_name}.ckpt'
    model_cls = models[model_name]
    if ckpt_path.exists() and model_name != 'MQF2' and not retrain:
        model = model_cls.load_from_checkpoint(str(ckpt_path))
    else:
        p, d = datamodule.input_dim, datamodule.output_dim
        model = model_cls(p, d, **kwargs)
        trainer = get_lightning_trainer(rc)
        trainer.fit(model, datamodule)
        trainer.save_checkpoint(ckpt_path)
    model.to(rc.config.device)
    return model


def get_oracle_model(rc, datamodule):
    oracle_model = OracleModel()
    #oracle_model.to(rc.config.device)
    trainer = DefaultTrainer(rc)
    trainer.fit(oracle_model, datamodule)
    return oracle_model
