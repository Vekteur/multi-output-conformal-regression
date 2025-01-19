from torch.distributions import TransformedDistribution


class OracleModel:
    """
    Oracle model that has knowledge of the data generating process.
    """
    def __init__(self):
        self.device = 'cpu'

    def fit(self, x, y):
        self.distribution_generator = self.trainer.datamodule.distribution_generator
        assert self.distribution_generator is not None

    def predict(self, x):
        x = self.trainer.datamodule.scaler_x.inverse_transform(x)
        dist = self.distribution_generator.dist_y(x)
        return TransformedDistribution(dist, self.trainer.datamodule.scaler_y.transformer)

    def to(self, device):
        # We do this to match the interface of other modules
        # but this is not necessary because there are no parameters
        self.device = device
        return self

    @classmethod
    def output_type(cls):
        return 'distribution'
