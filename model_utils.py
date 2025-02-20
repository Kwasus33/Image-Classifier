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
    batch_sizes = []
    all_labels = []
    all_preds = []
    accs = []

    with torch.no_grad():
        # images is tensor of RGB values, labels is tensor of indexes 0-99 (for cifar-100) indicating class of image
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            batch_sizes.append(images.shape[0])
            acc = torch.mean((torch.argmax(outputs, dim=1) == labels).float()).item()
            accs.append(acc)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    loss = running_loss / len(loader)
    accuracy = np.average(accs, weights=batch_sizes)
    precision = precision_score(
        y_true=all_labels, y_pred=all_preds, average="macro", zero_division=1
    )
    recall = recall_score(
        y_true=all_labels, y_pred=all_preds, average="macro", zero_division=1
    )
    f1 = f1_score(y_true=all_labels, y_pred=all_preds, average="macro", zero_division=1)
    metrics = (accuracy, precision, recall, f1)

    return loss, metrics, preds
