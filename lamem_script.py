from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, StochasticWeightAveraging

from lit_modules import LitVGG, LitLaMemDataModule
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    data_path = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/preprocessed/dataset"

    # TODO: Change the root to data_path
    lamem = LitLaMemDataModule(
        root=data_path, batch_size=128, num_workers=5, dev_mode=False
    )
    # cifar = CifarDataModule()
    model = LitVGG(learning_rate=1e-5)

    tb_logger = TensorBoardLogger("./memorability")

    trainer = Trainer(
        max_epochs=500,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), DeviceStatsMonitor()],
        fast_dev_run=False,
        devices="auto",
        accelerator="gpu",
        num_nodes=1,
        strategy="ddp",
        # overfit_batches=1,
        gradient_clip_val=0.5,
        logger=tb_logger,
    )

    trainer.fit(model=model, datamodule=lamem)


if __name__ == "__main__":
    main()
