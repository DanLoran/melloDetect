import torch
from mellolib.globalConstants import PRETRAINED_MODEL_POOL

def validatePretrained(name):
    """
    validate whether a certain model is available in the pool of pretrained models
    """
    # check whether the model is available
    assert name in PRETRAINED_MODEL_POOL.keys()


def getPretrainedModelNoFc(name):
    """
    get a pretrained model with no classifier, just feature extraction,
    from the name.
    """

    # get model
    model = PRETRAINED_MODEL_POOL[name]

    # remove the final layer
    if 'resnet' in name:
        model.fc = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()

    # freeze the model
    for param in model.parameters():
        param.require_grad = False

    return model
