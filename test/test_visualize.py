from ..mellolib.visualize import ExtractFeatureLevel
import torchvision
import torch
import os
import pytest

def test_ExtractFeatureLevel():

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()

    cwd = os.getcwd()
    features = ExtractFeatureLevel(model, cwd)
    assert len(features) == 1

def test_ExtractFeatureLevelWrongFilePath():

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()

    with pytest.raises(FileNotFoundError):
        ExtractFeatureLevel(model, 'fakepath')
