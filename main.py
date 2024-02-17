from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor

from lit_modules import LitAlexNet, TinyImageNetDataModule


def main():
    data_path = "data/tiny-imagenet/tiny-imagenet-200/train"
    imagenet = TinyImageNetDataModule(data_path=data_path)
    model = LitAlexNet()

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min"), DeviceStatsMonitor()],
        fast_dev_run=False,
        devices=1,
        overfit_batches=1,
    )

    trainer.fit(model=model, datamodule=imagenet)


if __name__ == "__main__":
    main()
