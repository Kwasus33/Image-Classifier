import torch
from torch import nn

from classifier import GolemClassifier

from backbone import GolemBackbone
import torchvision.models as models
import os.path

from model_utils import train_epoch

PATH = "data/pretrained_models/"

RESNET18_NAME = "resnet18.pth"
RESNET34_NAME = "resnet34.pth"
VIT_NAME = "vit.pth"

EPOCH_COUNT = 50

def pretrain_resnet(classes_num, criterion, lr, optimizer, dataloader, device, model, model_name):
    # for param in model.parameters():
    #     param.requires_grad = False
    gc = GolemClassifier(model, model.fc.out_features, classes_num)
    gc.to(device)
    gc_criterion = nn.CrossEntropyLoss()
    gc_optimizer = optimizer(params=gc.parameters(), lr=lr)
    for i in range(EPOCH_COUNT):
        train_epoch(gc, gc_criterion, gc_optimizer, dataloader, device)
    torch.save(gc, PATH + model_name)

def load_model(classes_num, criterion, lr, optimizer, dataloader, device, model_name):
    if not os.path.isfile(PATH + model_name):
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        if model_name == RESNET18_NAME:
            pretrain_resnet18(classes_num, criterion, lr, optimizer, dataloader, device)
        elif model_name == RESNET34_NAME:
            pretrain_resnet34(classes_num, criterion, lr, optimizer, dataloader, device)
        else:
            pass
    model = torch.load(PATH + model_name, weights_only=False)
    model.to(device)
    return model

def get_golem_model(classes_num):
    bb = GolemBackbone()
    return GolemClassifier(bb, bb.fc.out_features, classes_num)

def get_resnet18_model(classes_num, criterion, lr, optimizer, dataloader, device):
    return load_model(classes_num, criterion, lr, optimizer, dataloader, device, RESNET18_NAME)

def pretrain_resnet18(classes_num, criterion, lr, optimizer, dataloader, device):
    resnet18 = models.resnet18(weights='IMAGENET1K_V1')
    pretrain_resnet(classes_num, criterion, lr, optimizer, dataloader, device, resnet18, RESNET18_NAME)

def get_resnet34_model(classes_num, criterion, lr, optimizer, dataloader, device):
    return load_model(classes_num, criterion, lr, optimizer, dataloader, device, RESNET34_NAME)

def pretrain_resnet34(classes_num, criterion, lr, optimizer, dataloader, device):
    resnet34 = models.resnet34(weights='IMAGENET1K_V1')
    pretrain_resnet(classes_num, criterion, lr, optimizer, dataloader, device, resnet34, RESNET34_NAME)

def get_vitbase_model(classes_num):
    vitbase = models.vit_b_16()
    return GolemClassifier(vitbase, vitbase.heads.head.out_features, classes_num)
