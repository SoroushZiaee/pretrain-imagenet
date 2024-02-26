import lightning as L
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader

import torch


class CifarDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")

    def setup(self, stage: str):
        # Define a transform to normalize the data
        # Load CIFAR-10 dataset
        transform = transforms.Compose(
            [
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize for CIFAR-10
            ]
        )

        self.train_dataset = CIFAR10(
            root="/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data",
            train=True,
            download=True,
            transform=transform,
        )

        # Load CIFAR10 test set
        self.test_dataset = CIFAR10(
            root="/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data",
            train=False,
            download=True,
            transform=transform,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=False,
        )
