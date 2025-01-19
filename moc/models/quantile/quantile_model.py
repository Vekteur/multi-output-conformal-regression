import logging

import torch
from lightning.pytorch import LightningModule


log = logging.getLogger('moc')


def quantile_scores(y, quantiles, alpha):
    """
    Return quantile scores for a tensor of quantile levels `alpha`.
    `y` is a tensor of shape (..., d)
    `quantiles` is a tensor of shape (..., d, n_levels)
    `alpha` is a tensor of shape (..., n_levels)
    """
    d, n_levels = quantiles.shape[-2:]
    assert alpha.shape[-1] == n_levels and y.shape[-1] == d
    diff = y[..., None] - quantiles
    assert diff.shape[-2:] == (d, n_levels)
    indicator = (diff < 0).float()
    score_per_level = diff * (alpha - indicator)
    return score_per_level


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                input_dim if i == 0 else hidden_size,
                hidden_size if i != num_layers - 1 else output_dim,
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class QuantileModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: torch.Tensor = None,
        hidden_size: int = 100,
        num_layers: int = 3,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.alpha.dim() == 1
        self.output_dim = output_dim
        self.model = MLP(
            input_dim=input_dim,
            output_dim=self.hparams.alpha.shape[0] * self.output_dim,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
        )

    def forward(self, x):
        out = self.model(x)
        out = out.unflatten(-1, (self.output_dim, self.hparams.alpha.shape[0]))
        return out
    
    def predict(self, x):
        return self(x).detach()

    def compute_loss(self, quantiles, y):
        return quantile_scores(y, quantiles, self.hparams.alpha).mean()

    def step(self, batch):
        x, y = batch
        quantiles = self(x)
        loss = self.compute_loss(quantiles, y)
        #print('Loss:', loss, flush=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    @classmethod
    def output_type(cls):
        return 'quantile'
