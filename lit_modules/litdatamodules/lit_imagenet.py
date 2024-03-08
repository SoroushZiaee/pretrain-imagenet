import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.ImageNet.ImageNetDataset import ImageNet


class LitImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        meta_path: str,
        num_workers: int = 10,
        batch_size: int = 32,
        desired_image_size: int = 224,
    ) -> None:
        super().__init__()

        self.root = root
        self.meta_path = meta_path
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.save_hyperparameters("batch_size")
        self.save_hyperparameters("desired_image_size")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_tranformers = transforms.Compose(
            [
                transforms.RandomResizedCrop(desired_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transformers = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(desired_image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def setup(self, stage: str) -> None:
        self.train_dataset = ImageNet(
            root=self.root,
            split="train",
            dst_meta_path=self.meta_path,
            transform=self.train_tranformers,
        )

        self.val_dataset = ImageNet(
            root=self.root,
            split="validation",
            dst_meta_path=self.meta_path,
            transform=self.val_transformers,
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
