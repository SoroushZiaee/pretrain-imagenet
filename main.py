from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, StochasticWeightAveraging

from lit_modules import LitAlexNet, TinyImageNetDataModule, CifarDataModule
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    data_path = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/tiny-imagenet/tiny-imagenet-200/train"
    imagenet = TinyImageNetDataModule(data_path=data_path, batch_size=96)
    # cifar = CifarDataModule()
    model = LitAlexNet(
        learning_rate=1e-3, num_classes=1000, example_input_array=(96, 3, 224, 224)
    )

    tb_logger = TensorBoardLogger(".")

    trainer = Trainer(
        max_epochs=500,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        fast_dev_run=False,
        devices="auto",
        accelerator="gpu",
        num_nodes=1,
        strategy="auto",
        # overfit_batches=1,
        gradient_clip_val=0.5,
        logger=tb_logger,
    )

    trainer.fit(model=model, datamodule=imagenet)


if __name__ == "__main__":
    main()
