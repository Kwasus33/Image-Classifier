from classifier import GolemClassifier

from backbone import GolemBackbone
import torchvision.models as models


def get_golem_model(classes_num):
    bb = GolemBackbone()
    return GolemClassifier(bb, bb.fc.out_features, classes_num)


def get_resnet18_model(classes_num):
    resnet18 = models.resnet18()
    return GolemClassifier(resnet18, resnet18.fc.out_features, classes_num)


def get_resnet34_model(classes_num):
    resnet34 = models.resnet34()
    return GolemClassifier(resnet34, resnet34.fc.out_features, classes_num)


def get_vitbase_model(classes_num):
    vitbase = models.vit_b_16()
    return GolemClassifier(vitbase, vitbase.heads.head.out_features, classes_num)
