import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse

from src.backbone import GolemBackbone
from src.classifier import GolemClassifier
from src.dataloaders import get_dataloaders
from src.model_utils import train_epoch, eval

from model_factories import (
    get_golem_model,
    get_resnet18_model,
    get_resnet34_model,
    get_vitbase_model,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    return parser.parse_args()


def main():
    train_loader, test_loader, classes = get_dataloaders()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parse_arguments()

    if args.BACKBONE == "resnet18":
        gc = get_resnet18_model()
    elif args.BACKBONE == "resnet34":
        gc = get_resnet34_model()
    elif args.BACKBONE == "ViT":
        gc = get_vitbase_model()
    else:
        gc = get_golem_model()

    criterion = (
        nn.CrossEntropyLoss() if args.LOSS == "CrossEntropy" else nn.MSELoss()
    )  # if args.LOSS == "CrossEntropy" else RMSE

    optimizer = (
        optim.SGD(gc.parameters(), lr=1e-3)
        if args.OPTIM == "sgd"
        else (
            optim.RMSprop(gc.parameters(), lr=1e-3)
            if args.OPTIM == "rms"
            else (
                optim.Adadelta(gc.parameters(), lr=1e-3)
                if args.OPTIM == "adadelta"
                else (
                    optim.Adamax(gc.parameters(), lr=1e-3)
                    if args.OPTIM == "adamax"
                    else optim.Adam(gc.parameters(), lr=1e-3)
                )
            )
        )
    )

    for i in range(2):
        loss = train_epoch(gc, criterion, optimizer, train_loader, device)
        print(f"{i+1}/2: loss={loss}")


if __name__ == "__main__":
    main()
