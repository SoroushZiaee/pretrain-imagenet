import lightning as L
import torch
from torch import nn


class LitAlexNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.rand(size=(64, 3, 224, 224))
        self.alex_net = torch.hub.load(
            "pytorch/vision:v0.10.0", "alexnet", weights=None
        )

        # Define criterion (loss function)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.alex_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.alex_net(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.alex_net(x)
        loss = self.criterion(output, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=100)
        return optimizer
