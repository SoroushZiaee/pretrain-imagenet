from typing import Tuple
import lightning as L
import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision.models import AlexNet_Weights
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau


import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # Default is for ImageNet
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.features[3].bias, 1)  # Second conv layer
        nn.init.constant_(self.features[8].bias, 1)  # Fourth conv layer
        nn.init.constant_(self.features[10].bias, 1)  # Fifth conv layer

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)


class LitAlexNet(L.LightningModule):
    def __init__(
        self,
        learning_rate,
        num_classes: int = 1000,
        example_input_array: Tuple[int] = (64, 3, 224, 224),
    ):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.save_hyperparameters("num_classes")

        self.example_input_array = torch.rand(size=example_input_array)

        # Load the pretrained AlexNet model
        self.alex_net = AlexNet(num_classes=self.hparams.num_classes)

        # self.alex_net = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "alexnet", weights=None
        # )

        print(f"everything is loaded.")

        # Define criterion (loss function) and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_top_1 = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes, top_k=1
        )
        self.accuracy_top_5 = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes, top_k=5
        )

    def forward(self, x):
        return self.alex_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # assert y.min() >= 0 and y.max() < 10, "Label indices should be within [0, 9]."
        output = self.alex_net(x)
        loss = self.criterion(output, y)
        # Calculate Top-1 accuracy
        top1_acc = self.accuracy_top_1(output, y)
        # Calculate Top-1 error
        top1_error = 1 - top1_acc

        top5_acc = self.accuracy_top_5(output, y)

        top5_error = 1 - top5_acc

        # Log Top-1 error to TensorBoard
        self.log(
            "train_top1_error",
            top1_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log Top-1 error to TensorBoard
        self.log(
            "train_top5_error",
            top5_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "Learning Rate",
            lr,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.alex_net(x)
        loss = self.criterion(output, y)
        # Calculate Top-1 accuracy
        top1_acc = self.accuracy_top_1(output, y)
        # Calculate Top-1 error
        top1_error = 1 - top1_acc

        top5_acc = self.accuracy_top_5(output, y)

        top5_error = 1 - top5_acc

        # Log Top-1 error to TensorBoard
        self.log(
            "val_top1_error",
            top1_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log Top-1 error to TensorBoard
        self.log(
            "val_top5_error",
            top5_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )

    def configure_optimizers(self):
        # Define your optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )

        return optimizer

    def on_epoch_end(self):
        # Log the learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.experiment.add_scalar("Learning Rate", lr, self.current_epoch)
        self.log("Learning Rate", lr, prog_bar=True, on_step=True, on_epoch=True)

    def on_after_backward(self):
        if self.trainer.global_step % 25 == 0:  # Log every 25 steps
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    name, param, self.trainer.global_step
                )
                if param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}_grad", param.grad, self.trainer.global_step
                    )
