import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import itertools
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


def getBestOptimLoss():
    # Loss functions
    criterions = [
        nn.CrossEntropyLoss,
        # CrossEntropy is meant to be used in multiclass classification, automatically deals with different size tensors
        # nn.MSELoss --> to use MSELoss we need to set same size of outputs (num_of_classes = 10 or 100) and labels (batch_size = 32)
    ]

    # Optimizers
    optimizers = [
        optim.Adam,
        optim.SGD,
        optim.RMSprop,
        #   optim.Adadelta, optim.Adamax
    ]

    combinations = list(itertools.product(criterions, optimizers))

    return combinations


def getBestModelParams(model, loader, device):
    min_loss = np.inf
    best_lr = 0
    best_optim = None
    best_lossFunc = None

    combinations = getBestOptimLoss()
    # Finds optimal learning rate for most popular CNN funcs:
    for lossFunc, optim in combinations:
        print(f"Test for optim {optim} and loss func {lossFunc}: ")
        for lr in np.array([0.0001, 0.001, 0.01, 0.1]):
            print(f"Learning rate: {lr}")
            criterion = lossFunc()
            optimizer = optim(params=model.parameters(), lr=lr)
            losses = []

            for i in range(3):
                loss = train_epoch(model, criterion, optimizer, loader, device)
                print(f"{i+1}/3: loss={loss}")
                if (i != 0 and loss > losses[-1]) or np.isnan(loss):
                    break
                losses.append(loss)

            if len(losses) != 0 and losses[-1] < min_loss:
                best_lr = lr
                min_loss = losses[-1]
                best_lossFunc = lossFunc
                best_optim = optim

    return best_lr, min_loss, best_lossFunc, best_optim
