from pathlib import Path

import yaml
import torch
from larsflow import Glow


class GlowDistribution(torch.distributions.Distribution):
    def __init__(self, glow, x, validate_args=False):
        self.glow = glow
        self.x = x
        super().__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )
    
    def log_prob(self, y):
        sample_shape = y.shape[:-(len(self.batch_shape) + 1)]
        x_repeat = self.x.view((1,) * len(sample_shape) + self.x.shape).expand(sample_shape + self.x.shape)
        y_flat, x_repeat_flat = y.reshape((-1,) + self.event_shape), x_repeat.reshape(-1)
        logp = self.glow.log_prob(y_flat, x_repeat_flat)
        return logp.reshape(sample_shape + self.batch_shape)
    
    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        x_repeat = self.x.repeat_interleave(
            repeats=sample_shape.numel(), dim=0
        )
        # num_samples is ignored when y is provided
        y, _ = self.glow.sample(num_samples=-1, y=x_repeat.flatten(), temperature=1.)

        y = torch.clamp(y, 0., 1.)
        y[torch.isnan(y)] = 0.

        y = y.reshape(
            (self.batch_shape.numel(), sample_shape.numel(), self.event_shape.numel())
        )
        y = y.reshape(sample_shape + self.batch_shape + (self.event_shape.numel(),))
        return y

    @property
    def batch_shape(self) -> torch.Size:
        return self.x.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size(self.glow.output_shape)


class GlowPreTrained(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_shape = (1,)
        self.output_shape = (3, 32, 32)
        self.num_classes = 10
    
    def fit(self, x, y):
        assert self.trainer.datamodule.dataset == 'cifar10'

        path = Path('models/glow_gauss_32')
        with open(path / 'glow_gauss_32.yaml', 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)
        config['model']['input_shape'] = self.output_shape
        config['model']['num_classes'] = self.num_classes

        self.model = Glow(config['model'])
        self.model.load_state_dict(torch.load(path / 'glow_gauss_32.pth', map_location=torch.device('cpu')))
        self.model.output_shape = self.output_shape
    
    def forward(self, x):
        x = x.to(torch.int64)
        return GlowDistribution(self.model, x)
    
    def predict(self, x):
        return self(x)
    
    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def output_type(cls):
        return 'distribution'
