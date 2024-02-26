import lightning as L
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch


class TinyImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size=32) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.data_path = data_path

    def setup(self, stage: str):
        transformers = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        full_dataset = ImageFolder(root=self.data_path, transform=transformers)
        # Split into training (80% and testing (20%) datasets)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=20,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=20,
            shuffle=False,
        )
