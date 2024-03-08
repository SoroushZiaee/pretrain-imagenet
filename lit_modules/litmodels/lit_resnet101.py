import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision.models import resnet101
import lightning as L
import torch.optim as optim


class ResNet101Regression(nn.Module):
    def __init__(self, num_output_features=1, task: str = "regression"):
        super().__init__()
        # Load a pre-trained ResNet-50 model
        self.resnet = resnet101(weights=None)
        self.task = task
        # Replace the classifier layer for regression

        if task == "regression":
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_output_features)
            # If your targets are in the range [0, 1], you might want to add a sigmoid layer:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)

        if self.task == "regression":
            x = self.sigmoid(x)

        return x


class LitResNet101(L.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        num_output_features: int = 1,
        example_input_array=(64, 3, 224, 224),
        task: str = "regression",
    ):
        super().__init__()
        self.model = ResNet101Regression(
            num_output_features=num_output_features, task=task
        )
        self.learning_rate = learning_rate
        self.task = task
        self.save_hyperparameters("learning_rate")

        self.example_input_array = torch.rand(size=example_input_array)

        if task == "regression":
            self.criterion = nn.MSELoss()

        else:
            self.criterion = nn.CrossEntropyLoss()
            self.top1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)
            self.top5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)

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
        output = self.model(x)

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
        output = self.model(x)
        memnet_output = self.test_memnet(x)

        loss = self.criterion(output.squeeze(), memnet_output.squeeze())
        self.log("test_loss", loss, prog_bar=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )

        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = self.hparams.learning_rate
        # Assuming a decay factor of gamma (e.g., 0.1) every 150 epochs

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
            last_epoch=39,
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
