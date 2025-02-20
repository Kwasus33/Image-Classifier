import torch
import torch.nn as nn

# import torch.optim as optim
import torchvision.models as models
import os.path

from classifier import GolemClassifier
from backbone import GolemBackbones
from model_utils import train_epoch

from enum import Enum


PATH = "data/pretrained_models/"

RESNET18_NAME = "resnet18"
RESNET34_NAME = "resnet34"
VIT_NAME = "vitbase"
GOLEM_BB1_NAME = "golem_bb1"
GOLEM_BB2_NAME = "golem_bb2"
GOLEM_BB3_NAME = "golem_bb3"


EPOCH_COUNT = 50


class Model(Enum):
    RESNET18 = 1
    RESNET34 = 2
    VITBASE = 3
    GOLEM_BB1 = 4
    GOLEM_BB2 = 5
    GOLEM_BB3 = 6


def train_model(
    model, criterion, optimizer, lr, is_lp, dataloader, classes_num, device
):
    """
    trains model with given params and saves it
    """
    if model == Model.RESNET18:
        gc = models.resnet18(weights="IMAGENET1K_V1")
    elif model == Model.RESNET34:
        gc = models.resnet34(weights="IMAGENET1K_V1")
    elif model == Model.VITBASE:
        gc = models.vit_b_16(weights="IMAGENET1K_V1")
    elif model == Model.GOLEM_BB1:
        gc = GolemClassifier(GolemBackbones.GM1, classes_num)
    elif model == Model.GOLEM_BB2:
        gc = GolemClassifier(GolemBackbones.GM2, classes_num)
    else:
        gc = GolemClassifier(GolemBackbones.GM3, classes_num)

    if model in (Model.RESNET18, Model.RESNET34, Model.VITBASE):
        if is_lp:
            for param in gc.parameters():
                param.requires_grad = False
        if model == Model.VITBASE:
            gc.heads.head = nn.Linear(gc.heads.head.in_features, classes_num)
        else:
            gc.fc = nn.Linear(gc.fc.in_features, classes_num)

    gc.to(device)
    gc_criterion = criterion()
    gc_optimizer = optimizer(params=gc.parameters(), lr=lr)

    for i in range(EPOCH_COUNT):
        loss = train_epoch(gc, gc_criterion, gc_optimizer, dataloader, device)
        print(f"Epoch {i}/{EPOCH_COUNT} loss: {loss}\n")

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    torch.save(
        gc,
        f"{PATH}{get_model_file_name(model, is_lp)}_lr_{lr}_classes_num_{classes_num}",
    )


def load_model(classes_num, criterion, lr, optimizer, dataloader, device, model, is_lp):
    """
    loads model with given params if already trained and saved,
    otherwise trains model with given params and saves it
    """
    model_file = (
        f"{PATH}{get_model_file_name(model, is_lp)}_lr_{lr}_classes_num_{classes_num}"
    )
    if not os.path.isfile(model_file):
        train_model(
            model, criterion, optimizer, lr, is_lp, dataloader, classes_num, device
        )
    model = torch.load(model_file, weights_only=False)
    model.to(device)
    return model


def get_model_file_name(model, is_lp):
    if model == Model.RESNET18:
        name = RESNET18_NAME
    elif model == Model.RESNET34:
        name = RESNET34_NAME
    elif model == Model.VITBASE:
        name = VIT_NAME
    elif model == Model.GOLEM_BB1:
        name = GOLEM_BB1_NAME
    elif model == Model.GOLEM_BB2:
        name = GOLEM_BB2_NAME
    else:
        name = GOLEM_BB3_NAME

    if is_lp:
        name += "_linear_probing"
    else:
        name += "_fine_tuning"

    return name
