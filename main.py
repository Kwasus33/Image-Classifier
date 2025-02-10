import torch.nn as nn
import torch.optim as optim
import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from classifier import GolemClassifier


def get_dataloaders():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # transforms imgs to tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # new values: x' = (x−μ)/σ
            ),  # normalizes data - mean and std are set to 0.5 for R, G, B
        ]
    )

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # creates iterable set of batch sets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, testloader, classes

def train_epoch(model: nn.Module, criterion, optimizer, loader: torch.utils.data.DataLoader, device: torch.device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss

def eval(model: nn.Module, criterion, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

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
