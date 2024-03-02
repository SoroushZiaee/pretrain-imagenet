from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    StochasticWeightAveraging,
    LearningRateMonitor,
)

from lit_modules import LitLaMemDataModule, LitVGG19
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    data_path = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/lamem_images/lamem/"

    # TODO: Change the root to data_path
    lamem = LitLaMemDataModule(
        root=data_path, batch_size=128, num_workers=5, dev_mode=False
    )
    model = LitVGG19(learning_rate=1.2022644346174132e-06)

    tb_logger = TensorBoardLogger("./vgg19_lalem")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=500,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        callbacks=[DeviceStatsMonitor(), lr_monitor],
        fast_dev_run=False,
        devices="auto",
        accelerator="gpu",
        num_nodes=1,
        strategy="ddp_find_unused_parameters_true",
        # overfit_batches=1,
        gradient_clip_val=0.5,
        logger=tb_logger,
    )

    trainer.fit(model=model, datamodule=lamem)


if __name__ == "__main__":
    main()
