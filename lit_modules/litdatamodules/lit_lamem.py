import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision import transforms
import PIL.Image
import os
from torch.utils.data import DataLoader
import numpy as np

from datasets.LaMem.LaMemDataset import LaMem


class LitLaMemDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        num_workers: int = 10,
        batch_size: int = 32,
        desired_image_size: int = 224,
        dev_mode: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")

        self.root = root
        self.num_workers = num_workers
        self.dev_mode = dev_mode
        self.desired_image_size = desired_image_size

        splits_list = os.listdir(os.path.join(root, "splits"))
        self.train_splits = list(sorted(filter(lambda x: "train" in x, splits_list)))
        self.val_splits = list(sorted(filter(lambda x: "val" in x, splits_list)))
        self.test_splits = list(sorted(filter(lambda x: "test" in x, splits_list)))

        # Use one Split train_1.csv
        self.train_splits = [self.train_splits[0]]
        self.val_splits = [self.val_splits[0]]
        self.test_splits = [self.test_splits[0]]

        self.mean = np.load(
            "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/datasets/LaMem/support_files/image_mean.npy"
        )

    def setup(self, stage: str) -> None:

        if self.dev_mode:
            transforms_list = transforms.Compose(
                [
                    transforms.Resize((256, 256), PIL.Image.BILINEAR),
                    # lambda x: np.array(x),
                    # lambda x: np.subtract(
                    #     x[:, :, [2, 1, 0]], self.mean
                    # ),  # Subtract average mean from image (opposite order channels)
                    # lambda x: x[15:242, 15:242],  # Center crop
                    transforms.ToTensor(),
                ]
            )

        else:
            transforms_list = transforms.Compose(
                [
                    transforms.Resize((256, 256), PIL.Image.BILINEAR),
                    lambda x: np.array(x),
                    lambda x: np.subtract(
                        x[:, :, [2, 1, 0]], self.mean
                    ),  # Subtract average mean from image (opposite order channels)
                    transforms.ToTensor(),
                    transforms.CenterCrop(
                        self.desired_image_size
                    ),  # Center crop to 224x224
                ]
            )

        self.train_dataset = LaMem(
            root=self.root, splits=self.train_splits, transforms=transforms_list
        )

        self.val_dataset = LaMem(
            root=self.root, splits=self.val_splits, transforms=transforms_list
        )

        self.test_dataset = LaMem(
            root=self.root, splits=self.test_splits, transforms=transforms_list
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=2,
        )
