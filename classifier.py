import torch.nn as nn


class GolemClassifier(nn.Module):
    def __init__(self, model, features, classes_num):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(features, classes_num)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
