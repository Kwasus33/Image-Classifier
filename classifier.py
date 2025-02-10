
import torch.nn as nn

class GolemClassifier(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.fc = nn.Linear(features, 100)
    
    def forward(self, x):
        return self.fc(x)

