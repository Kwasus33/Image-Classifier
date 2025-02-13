import torch.nn as nn
import torch


def train_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    loader: torch.utils.data.DataLoader,
    # device: torch.device,
):
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


def eval(
    model: nn.Module,
    criterion,
    loader: torch.utils.data.DataLoader,
    # device: torch.device,
):
    model.eval()
    running_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_correct += (outputs == labels).float().sum()

    return (running_loss / len(loader)), (total_correct / total_samples)
