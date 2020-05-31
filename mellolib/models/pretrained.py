import torchvision
import torch

global pretrained_model_pool
pretrained_model_pool = {
    'resnet18': torchvision.models.resnet18(pretrained=True),
    'resnet34': torchvision.models.resnet34(pretrained=True),
    'resnet50': torchvision.models.resnet50(pretrained=True),
    'resnet101': torchvision.models.resnet101(pretrained=True),
    'alexnet': torchvision.models.alexnet(pretrained=True),
}

"""
validate whether a certain model is available in the pool of pretrained models
"""
def validatePretrained(name):
    # check whether the model is available
    assert name in pretrained_model_pool.keys()

"""
get a pretrained model with no classifier, just feature extraction,
from the name.
"""
def getPretrainedModelNoFc(name):

    # get model
    model = pretrained_model_pool[name]

    # remove the final layer
    model.fc = torch.nn.Identity()

    # freeze the model
    for param in model.parameters():
        param.require_grad = False;

    return model
