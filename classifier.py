import torch.nn as nn

from backbone import *


class GolemClassifier(nn.Module):
    def __init__(self, model, classes_num):
        super().__init__()
        if model == GolemBackbones.GM1:
            self.model = GolemBackbone()
        else:
            self.model = GolemBackbone2()
        self.fc = nn.Linear(1000, classes_num)
        self.fc.weight = nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        self.fc.bias = nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
