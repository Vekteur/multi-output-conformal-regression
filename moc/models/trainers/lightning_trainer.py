from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)

class CustomLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs.item()
        self.val_losses.append(loss)


def get_lightning_trainer(rc):
    # Might be better to do this in the future:
    # Rule of thumb:
    # - Validation set of size val_size = min(2500, 0.1 * n)
    #   - The obtained estimation of the loss is unbiased and is already precise with 2500 samples
    # - Test set of size test_size = 0.2 * n because this measures the metrics we are interested in
    #   - Having more samples is useful e.g. to measure conditional coverage
    # - Measure validation loss every 4 * val_size // batch_size steps
    #   - We can afford to measure validation during one fifth of the training
    # - Patience of 15

    ckpt = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1,  # save k best models (determined by above metric)
        save_last=False,  # save model from last epoch
        verbose=False,
        dirpath=str(rc.checkpoints_path),
        filename='epoch_{epoch:04d}',
        auto_insert_metric_name=False,
    )

    es = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=15,
        min_delta=0,
    )

    callbacks = [ckpt, es, CustomLogger()]

    accelerator = {
        'cpu': 'cpu',
        'cuda': 'gpu',
    }[rc.config.device]

    return Trainer(
        accelerator=accelerator,
        devices=1,
        min_epochs=1,
        max_epochs=2 if rc.config.fast else 5000,
        # number of validation steps to execute at the beginning of the training
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=False,
    )
