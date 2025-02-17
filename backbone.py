import torch.nn as nn

"""
needs adding proper activation funcs
"""


class GolemBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )  # in_channels=3 cause data(images) is represeted by 3 values (RGB)
        # batch size is set to 32
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 16 * 16, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(
            x.size(0), -1
        )  # tensor must be flatten before passing to fully connected layer
        return self.fc(x)
