import torch.nn as nn


class GolemClassifier(nn.Module):
    def __init__(self, model, features, classes_num):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(features, classes_num)
        self.fc.weight = nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        self.fc.bias = nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
