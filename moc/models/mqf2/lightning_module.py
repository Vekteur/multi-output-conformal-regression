# Adapted from https://github.com/awslabs/gluonts/tree/dev/src/gluonts/torch/model/mqf2

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
from torch.distributions import Normal, Independent
from lightning.pytorch import LightningModule
from cpflows.flows import ActNorm
from cpflows.icnn import PICNN

from .model import MyDeepConvexFlow, MySequentialFlow
from .distribution import MQF2Distribution
from moc.metrics.distribution_metrics import energy_score


def log_standard_normal(x):
    import numpy as np
    z = - 0.5 * float(np.log(2 * np.pi))
    return - x ** 2 / 2 + z


class MQF2Module(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        threshold_input: float = 100,
        is_energy_score: bool = False,
        estimate_logdet: bool = False,
    ) -> None:
        r"""
        Model class for the model MQF2 proposed in the paper ``Multivariate
        Quantile Function Forecaster`` by Kan, Aubet, Januschowski, Park,
        Benidis, Ruthotto, Gasthaus.

        The original implementation is modified to support tabular datasets
        """
        super().__init__()

        self.threshold_input = threshold_input
        self.is_energy_score = is_energy_score

        picnn = PICNN(
            dim=output_dim,
            dimh=icnn_hidden_size,
            dimc=input_dim,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexflow = MyDeepConvexFlow(
            picnn,
            output_dim,
            is_energy_score=is_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if is_energy_score:
            networks = [deepconvexflow]
        else:
            networks = [
                ActNorm(output_dim),
                deepconvexflow,
                ActNorm(output_dim),
            ]

        self.flow = MySequentialFlow(networks, output_dim)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def latent_dist(self, batch_shape):
        shape = batch_shape + (self.flow.dim,)
        return Independent(Normal(
            torch.zeros(shape, device=self.device),
            torch.ones(shape, device=self.device),
        ), 1)

    def forward(self, z, x):
        """
        `z` is of shape (sample_size, batch_size, dim)
        `x` is of shape (batch_size, dim)
        """
        z = z.permute(1, 0, 2)
        batch_shape, sample_shape = torch.Size((z.shape[0],)), torch.Size((z.shape[1],))
        z = z.reshape(-1, self.flow.dim)
        x_repeat = x.repeat_interleave(
            repeats=sample_shape.numel(), dim=0
        )
        if self.is_energy_score:
            y = self.flow(z, context=x_repeat)
        else:
            y = self.flow.reverse(z, context=x_repeat)

        y = y.reshape(
            (batch_shape.numel(), sample_shape.numel(), self.flow.dim,)
        )
        y = y.permute(1, 0, 2)
        y = y.reshape(sample_shape + batch_shape + (self.flow.dim,))
        return y

    def log_prob(self, y, x):
        batch_shape = x.shape[:-1]
        sample_shape = y.shape[:-(len(batch_shape) + 1)]
        x_repeat = x.view((1,) * len(sample_shape) + x.shape).expand(sample_shape + x.shape)
        y_flat, x_repeat_flat = y.reshape(-1, y.shape[-1]), x_repeat.reshape(-1, x_repeat.shape[-1])
        if self.is_energy_score:
            z = self.flow.reverse(y_flat, context=x_repeat_flat)
            y, logdet = self.flow.forward_transform(z, context=x_repeat_flat)
            logp0 = log_standard_normal(z).sum(-1)
            logp = logp0 - logdet
        else:
            logp = self.flow.logp(y_flat, x_repeat_flat)
        return logp.reshape(sample_shape + batch_shape)
    
    def dist(self, x):
        return MQF2Distribution(
            self, 
            x, 
            threshold_input=self.threshold_input, 
            is_energy_score=self.is_energy_score
        )
    
    def set_estimate_log_det(self, estimate_logdet: bool):
        self.flow.estimate_logdet = estimate_logdet



class MQF2LightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        icnn_hidden_size: int = 40,
        icnn_num_layers: int = 2,
        is_energy_score: bool = False,
        threshold_input: float = 100,
        es_num_samples: int = 50,
        estimate_logdet: bool = False,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MQF2Module(
            input_dim=input_dim,
            output_dim=output_dim,
            icnn_hidden_size=self.hparams.icnn_hidden_size,
            icnn_num_layers=self.hparams.icnn_num_layers,
            is_energy_score=self.hparams.is_energy_score,
            threshold_input=self.hparams.threshold_input,
            estimate_logdet=self.hparams.estimate_logdet,
        )

    def forward(self, x):
        return self.model.dist(x)
    
    def predict(self, x):
        return self(x)
    
    def compute_loss(self, dist, y):
        if self.hparams.is_energy_score:
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples, rsample=True).mean()
        else:
            return -dist.log_prob(y).mean()

    def step(self, batch):
        x, y = batch
        dist = self(x)
        return self.compute_loss(dist, y)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(
            'train/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.set_estimate_log_det(False)
        loss = self.step(batch)
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.model.set_estimate_log_det(self.hparams.estimate_logdet)
        return loss
    
    def on_train_end(self):
        self.model.set_estimate_log_det(False)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    @classmethod
    def output_type(cls):
        return 'distribution'
