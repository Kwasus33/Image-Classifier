from classifier import GolemClassifier

from backbone import GolemBackbone
import torchvision.models as models


def get_golem_model():
    bb = GolemBackbone()
    return GolemClassifier(bb, bb.fc.out_features)

def get_resnet18_model():
    resnet18 = models.resnet18()
    return GolemClassifier(resnet18, resnet18.fc.out_features)

def get_resnet34_model():
    resnet34 = models.resnet34()
    return GolemClassifier(resnet34, resnet34.fc.out_features)

def get_vitbase_model():
    vitbase = models.vit_b_16()
    return GolemClassifier(vitbase, vitbase.heads.head.out_features)
