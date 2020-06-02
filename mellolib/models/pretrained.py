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


def validatePretrained(name):
    """
    validate whether a certain model is available in the pool of pretrained models
    """
    # check whether the model is available
    assert name in pretrained_model_pool.keys()


def getPretrainedModelNoFc(name):
    """
    get a pretrained model with no classifier, just feature extraction,
    from the name.
    """

    # get model
    model = pretrained_model_pool[name]

    # remove the final layer
    if 'resnet' in name:
        model.fc = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()

    # freeze the model
    for param in model.parameters():
        param.require_grad = False

    return model
