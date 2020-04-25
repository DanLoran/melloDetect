import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

def resnet18(transfer=True):
    model = torchvision.models.resnet18(pretrained=transfer)

    # Freeze the model if transfer learning
    if (transfer):
        for param in model.parameters():
            param.require_grad = False;

    # Remake the final layer
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=512, out_features=100)),
        ('relu1', F.relu()),
        ('fc2', nn.Linear(in_features=100, out_features=10)),
        ('relu2', F.relu()),
        ('fc3', nn.Linear(in_features=10, out_features=2)),
        ('output', nn.Softmax(dim=1))
    ]))

    model.fc = fc
    return model
