import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def resnet18(transfer=True):
    model = torchvision.models.resnet18(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=512, out_features=100)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=100, out_features=10)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))

    model.fc = fc
    return model

def mobilenet(transfer=True):
    model = torchvision.models.mobilenet_v2(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('do', nn.Dropout(p=0.2, inplace=False)),
        ('fc1', nn.Linear(in_features=1280, out_features=100)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=100, out_features=10)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))

    model.classifier = fc
    return model

def alexnet(transfer=True):
    model = torchvision.models.alexnet(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('do', nn.Dropout(p=0.5, inplace=False)),
        ('fc1', nn.Linear(in_features=9216, out_features=4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=4096, out_features=1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=1000, out_features=100)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(in_features=100, out_features=10)),
        ('relu4', nn.ReLU()),
        ('fc5', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))

    model.classifier = fc
    return model

def vgg(transfer=True):
    model = torchvision.models.vgg16(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('do', nn.Dropout(p=0.5, inplace=False)),
        ('fc1', nn.Linear(in_features=25088, out_features=5000)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=5000, out_features=1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=1000, out_features=100)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(in_features=100, out_features=10)),
        ('relu4', nn.ReLU()),
        ('fc5', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))
    model.classifier = fc
    return model

# Not readily runable
# def densenet(transfer=True):
#     model = torchvision.models.densenet161(pretrained=transfer)
#
#     # Freeze the model if transfer learning
#     if (transfer):
#         for param in model.parameters():
#             param.require_grad = False;
#
#     # Remake the final layer
#     fc = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(in_features=2208, out_features=100)),
#         ('relu1', nn.ReLU()),
#         ('fc2', nn.Linear(in_features=100, out_features=10)),
#         ('relu2', nn.ReLU()),
#         ('fc3', nn.Linear(in_features=10, out_features=2)),
#         ('output', nn.Softmax(dim=1))
#     ]))
#     model.classifier = fc
#     return model

# Not readily runable
# def inception(transfer=True):
#     model = torchvision.models.inception_v3(pretrained=transfer)
#
#     # Freeze the model if transfer learning
#     if (transfer):
#         for param in model.parameters():
#             param.require_grad = False;
#
#     # Remake the final layer
#     fc = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(in_features=2048, out_features=100)),
#         ('relu1', nn.ReLU()),
#         ('fc2', nn.Linear(in_features=100, out_features=10)),
#         ('relu2', nn.ReLU()),
#         ('fc3', nn.Linear(in_features=10, out_features=2)),
#         ('output', nn.Softmax(dim=1))
#     ]))
#     model.fc = fc
#     return model

def googlenet(transfer=True):
    model = torchvision.models.googlenet(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=1024, out_features=100)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=100, out_features=10)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))
    model.fc = fc
    return model

def shufflenet(transfer=True):
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;


    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=1024, out_features=100)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(in_features=100, out_features=10)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))
    model.fc = fc
    return model
