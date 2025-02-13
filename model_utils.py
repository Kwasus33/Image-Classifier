import torch.nn as nn
import torch
from sklearn.metrics import recall_score, precision_score, f1_score


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

    loss = running_loss / len(loader)
    accuracy = total_correct / total_samples
    precision = precision_score(y_true=labels, y_pred=outputs)
    recall = recall_score(y_true=labels, y_pred=outputs)
    f1 = f1_score(y_true=labels, y_pred=outputs)

    return loss, accuracy, precision, recall, f1
