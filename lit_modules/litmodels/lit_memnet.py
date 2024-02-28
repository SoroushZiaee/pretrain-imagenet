from typing import Any, Dict
from torch import nn
import lightning as L


class MemNet(nn.Module):
    def __init__(self):
        super(MemNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True
        )
        self.norm1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1)
        self.conv2 = nn.Conv2d(
            96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True
        )
        self.norm2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1)
        self.conv3 = nn.Conv2d(
            256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2
        )
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2
        )
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True
        )
        self.fc6 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.5)
        self.fc8_euclidean = nn.Linear(in_features=4096, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        x = self.fc8_euclidean(x)
        return x


class LitMemNet(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.1,
        checkpoint: Dict = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("learning_rate")

        self.model = MemNet()

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["state_dict"])

    def forward(self, x):
        return self.model(x)
