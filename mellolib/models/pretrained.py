import torch
import torchvision
from mellolib.globalConstants import PRETRAINED_MODEL_POOL

def validatePretrained(name):
    """
    validate whether a certain model is available in the pool of pretrained models
    """
    # check whether the model is available
    assert name in PRETRAINED_MODEL_POOL


def getPretrainedModelNoFc(name):
    """
    get a pretrained model with no classifier, just feature extraction,
    from the name.
    """

    # validate name of the model
    validatePretrained(name)

    # get correct model
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)

    # remove the final layer
    if 'resnet' in name:
        model.fc = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()

    # freeze the model
    for param in model.parameters():
        param.require_grad = False

    return model
