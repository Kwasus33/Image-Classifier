import torch
from torch import nn

from classifier import GolemClassifier

from backbone import GolemBackbone
import torchvision.models as models
import os.path

from model_utils import train_epoch

from enum import Enum

class Model(Enum):
    RESNET18 = 1
    RESNET34 = 2
    VITBASE = 3
    GOLEM = 4



PATH = "data/pretrained_models/"

RESNET18_NAME = "resnet18"
RESNET34_NAME = "resnet34"
VIT_NAME = "vitbase"
GOLEM_NAME = "golem"

EPOCH_COUNT = 50

def pretrain_model(classes_num, criterion, lr, optimizer, dataloader, device, model, is_lp):
    if model == Model.RESNET18:
        gc = models.resnet18(weights='IMAGENET1K_V1')
    elif model == Model.RESNET34:
        gc = models.resnet34(weights='IMAGENET1K_V1')
    elif model == Model.VITBASE:
        gc = models.vit_b_16(weights='IMAGENET1K_V1')
    elif model == Model.GOLEM:
        gc = GolemBackbone()
    else:
        gc = models.resnet18(weights='IMAGENET1K_V1')

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
        train_epoch(gc, gc_criterion, gc_optimizer, dataloader, device)
    torch.save(gc, PATH + get_model_file_name(model, is_lp))

def load_model(classes_num, criterion, lr, optimizer, dataloader, device, model, is_lp):
    model_file = PATH + get_model_file_name(model, is_lp)
    if not os.path.isfile(model_file):
        pretrain_model(classes_num, criterion, lr, optimizer, dataloader, device, model, is_lp)
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
    elif model == Model.GOLEM:
        name = GOLEM_NAME
    else:
        name = "__"

    if is_lp:
        name += "_linear_probing"
    else:
        name += "_fine_tuning"

    name += ".pth"
    return name