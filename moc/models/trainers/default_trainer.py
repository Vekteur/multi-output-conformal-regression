class DefaultTrainer:
    def __init__(self, rc, **kwargs):
        self.rc = rc

    def fit(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        model.trainer = self
        x, y = datamodule.data_train[:]
        return model.fit(x, y)

    def test(self, model, datamodule, **kwargs):
        return model.test(datamodule)
