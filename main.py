import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from classifier import GolemClassifier
from dataloaders import get_dataloaders
from model_utils import train_epoch, eval



def main():
    train_loader, test_loader, classes = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = models.resnet18()
    gc = GolemClassifier(resnet18, resnet18.fc.out_features)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gc.parameters(), lr=1e-3)

    for i in range(2):
        loss = train_epoch(gc, criterion, optimizer, train_loader, device)
        print(f"{i+1}/2: loss={loss}")


if __name__ == "__main__":
    main()
