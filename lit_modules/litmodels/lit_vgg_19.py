import torch
import torch.nn as nn

import lightning as L
from torchmetrics import Accuracy


class VGG19(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG19, self).__init__()
        self.features = self._make_layers(
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG19Regression(VGG19):
    def __init__(self, num_output_features=1, task: str = "regression"):
        super().__init__(num_classes=num_output_features, init_weights=True)
        # Replace the last layer for regression

        self.task = task
        if task == "regression":
            self.classifier[-1] = nn.Linear(4096, num_output_features)
            # If your targets are in the range [0, 1], you might want to add a sigmoid layer:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        if self.task == "regression":
            x = self.sigmoid(x)
        return x


class LitVGG19(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_output_features: int = 1,
        task: str = "regression",
    ):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.task = task
        self.example_input_array = torch.randn(8, 3, 256, 256)

        self.vgg = VGG19Regression(num_output_features=num_output_features, task=task)

        if task == "regression":
            self.criterion = nn.MSELoss()

        else:
            self.criterion = nn.CrossEntropyLoss()
            self.top1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)
            self.top5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

    def forward(self, x):
        return self.vgg(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.vgg(x)

        loss = self.criterion(output.squeeze(), y)
        self.log("training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.task == "classification":
            top1_err = 1 - self.top1(output.squeeze(), y)
            top5_err = 1 - self.top5(output.squeeze(), y)

            self.log(
                "training_top1_err",
                top1_err,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "training_top5_err",
                top5_err,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.vgg(x)

        loss = self.criterion(output.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.task == "classification":
            top1_err = 1 - self.top1(output.squeeze(), y)
            top5_err = 1 - self.top5(output.squeeze(), y)

            self.log(
                "validation_top1_err",
                top1_err,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation_top5_err",
                top5_err,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.vgg(x)
        loss = self.criterion(output.squeeze(), y)
        self.log("test_loss", loss, prog_bar=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
        # Assuming a decay factor of gamma (e.g., 0.1) every 150 epochs

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=33,
            gamma=0.1,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_after_backward(self):
        if self.trainer.global_step % 25 == 0:  # Log every 25 steps
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    name, param, self.trainer.global_step
                )

                if param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}_grad",
                        param.grad,
                        self.trainer.global_step,
                    )
