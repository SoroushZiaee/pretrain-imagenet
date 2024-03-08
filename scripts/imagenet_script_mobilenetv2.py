import sys
import os

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    StochasticWeightAveraging,
    LearningRateMonitor,
)

from lit_modules import LitImageNetDataModule, LitMobileNetV2
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy


def main():
    root = "/datashare/ImageNet/ILSVRC2012"
    meta_path = (
        "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet"
    )
    datamodule = LitImageNetDataModule(
        root=root,
        meta_path=meta_path,
        num_workers=3,
        batch_size=128,
        desired_image_size=224,
    )

    num_output_features = 1000
    model = LitMobileNetV2(
        learning_rate=0.001,
        num_output_features=num_output_features,
        task="classification",
    )
    strategy = DeepSpeedStrategy()
    tb_logger = TensorBoardLogger("./runs/imagenet/mobilenetv2")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=500,
        callbacks=[
            DeviceStatsMonitor(),
            lr_monitor,
            # EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
        fast_dev_run=False,
        devices=2,
        accelerator="cuda",
        num_nodes=1,
        strategy=strategy,
        # overfit_batches=1,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        logger=tb_logger,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
