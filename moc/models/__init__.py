from moc.models.trainers.default_trainer import DefaultTrainer
from moc.models.trainers.lightning_trainer import get_lightning_trainer

from moc.models.oracle.oracle_model import OracleModel
from moc.models.drf_kde.drf_kde import DRF_KDE
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.glow.glow import GlowPreTrained
from moc.models.mixture.mixture_model import MixtureLightningModule
from moc.models.quantile.quantile_model import QuantileModule


models = {
    'Oracle': OracleModel,
    'DRF-KDE': DRF_KDE,
    'MQF2': MQF2LightningModule,
    'Mixture': MixtureLightningModule,
    'Glow': GlowPreTrained,
    'Quantile': QuantileModule,
}

trainers = {
    'Oracle': DefaultTrainer,
    'DRF-KDE': DefaultTrainer,
    'MQF2': get_lightning_trainer,
    'Mixture': get_lightning_trainer,
    'Glow': DefaultTrainer,
    'Quantile': get_lightning_trainer,
}
