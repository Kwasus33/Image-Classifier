import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def train_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
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
    device: torch.device,
):
    model.eval()
    running_loss = 0
    total_correct = 0
    total_samples = 0
    numpy_labels = []
    numpy_preds = []

    with torch.no_grad():
        # images is tensor of RGB values, labels is tensor of indexes 0-99 indicating class of image
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            total_samples += images.size(0)
            total_correct += (
                torch.tensor([x == y for x, y in zip(preds, labels)]).float().sum()
            )

    numpy_labels.extend(labels.cpu().numpy())
    numpy_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(loader)
    accuracy = total_correct / total_samples
    precision = precision_score(
        y_true=numpy_labels, y_pred=numpy_preds, average="macro", zero_division=1
    )
    recall = recall_score(
        y_true=numpy_labels, y_pred=numpy_preds, average="macro", zero_division=1
    )
    f1 = f1_score(
        y_true=numpy_labels, y_pred=numpy_preds, average="macro", zero_division=1
    )
    metrics = (loss, accuracy, precision, recall, f1)

    return metrics, preds
