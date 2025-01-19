import pandas as pd
import numpy as np
drf_imported = True
try:
    from drf import drf
except ImportError:
    drf_imported = False
import torch
from torch.distributions import Normal, Independent, MixtureSameFamily, Categorical

def kde(w, y, bandwidth):
    return MixtureSameFamily(
        mixture_distribution=Categorical(probs=w),
        component_distribution=Independent(Normal(loc=y, scale=torch.full(y.shape, bandwidth, device=y.device)), 1)
    )


class DRF_KDE:
    def __init__(self, input_dim, output_dim):
        self.device = 'cpu'

    def fit(self, x, y):
        num_trees = 100 if self.trainer.rc.config.fast else 2000
        if not drf_imported:
            raise ImportError('drf is not installed. See the README for installation instructions.')
        self.model = drf(min_node_size=15, num_trees=num_trees, splitting_rule='FourierMMD')
        self.model.fit(x, y)
        self.y_train = y
        self.bandwidth = self.select_bandwidth()
    
    def select_bandwidth(self):
        x, y = self.trainer.datamodule.data_val[:]
        grid = np.logspace(-3, 2, 20)
        best_nll = float('inf')
        best_bandwidth = None
        for bandwidth in grid:
            dist = self.predict(x, bandwidth)
            nll = -dist.log_prob(y).mean()
            if nll < best_nll:
                best_nll = nll
                best_bandwidth = bandwidth
        return best_bandwidth

    def predict(self, x, bandwidth=None):
        if bandwidth is not None:
            self.bandwidth = bandwidth
        weights = self.model.predict(x.cpu()).weights
        # KDE for the conditional distribution using a mixture of multivariate normals
        weights = torch.tensor(weights, device=self.device)
        y = self.y_train.repeat(x.shape[0], 1, 1).to(self.device)
        return kde(weights, y, self.bandwidth)

    def to(self, device):
        self.device = device
        return self

    @classmethod
    def output_type(cls):
        return 'distribution'
