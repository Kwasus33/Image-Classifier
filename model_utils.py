import torch.nn as nn
import torch

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