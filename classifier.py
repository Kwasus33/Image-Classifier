
import torch.nn as nn

class GolemClassifier(nn.Module):
    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(features, 10)
    
    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

