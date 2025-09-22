import torch.nn as nn

from backbone import *


class GolemClassifier(nn.Module):
    """
    self.model is CNN backbone containing conv, pool, batch and linear layers + dropouts
    self.fc is last linear layer - last fully connected layer - CNN classificator
    """

    def __init__(self, model, classes_num):
        super().__init__()

        if model == GolemBackbones.GM1:
            self.model = GolemBackbone1()
        elif model == GolemBackbones.GM2:
            self.model = GolemBackbone2()
        else:
            self.model = GolemBackbone3()

        fc_in = self.model.fc.out_features
        self.fc = nn.Linear(fc_in, classes_num)

        self.fc.weight = nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        self.fc.bias = nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
