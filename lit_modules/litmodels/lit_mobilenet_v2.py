import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import lightning as L


class MobileNetV2Regression(nn.Module):
    def __init__(self, num_output_features=1):
        super().__init__()
        # Load a pre-trained ResNet-50 model
        self.mobilenet = mobilenet_v2(weights=None)
        # Replace the classifier layer for regression

        self.mobilenet.classifier = nn.Sequential(
            nn.Flatten(
                start_dim=1
            ),  # Flatten [batch_size, channels, 1, 1] to [batch_size, channels]
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_output_features, bias=True),
        )
        # If your targets are in the range [0, 1], you might want to add a sigmoid layer:
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get the main output from the GoogLeNet model
        outputs = self.mobilenet(x)
        if isinstance(outputs, torch.Tensor):
            x = outputs
        else:  # If outputs are GoogLeNetOutputs, extract the main output
            x = outputs.logits

        # Apply the sigmoid function to the main output
        x = self.sigmoid(x)
        return x


class LitMobileNetV2(L.LightningModule):
    def __init__(self, learning_rate=1e-3, example_input_array=(64, 3, 224, 224)):
        super().__init__()
        self.model = MobileNetV2Regression(num_output_features=1)
        self.learning_rate = learning_rate
        self.save_hyperparameters("learning_rate")

        self.example_input_array = torch.rand(size=example_input_array)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)

        loss = self.criterion(output.squeeze(), y)
        self.log("training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)

        loss = self.criterion(output.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)

        loss = self.criterion(output.squeeze(), y)
        self.log("test_loss", loss, prog_bar=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=5e-4
        )
        # Assuming a decay factor of gamma (e.g., 0.1) every 150 epochs

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=200,
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
