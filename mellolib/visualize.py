import torchvision
import torch
import argparse
from PIL import Image
import os
from os import path
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt

"""
model: a pytorch model
imageDir: a string indicating where the images are located
"""
def ExtractFeatureLevel(model, imageDir):

    # in evaluation mode
    model.eval()
    for param in model.parameters():
        param.require_grad = False

    # transform and load the data
    transform = Compose([Resize((256, 256)), ToTensor()])
    evalset = torchvision.datasets.ImageFolder(
        root=imageDir, transform=transform)
    evalloader = torch.utils.data.DataLoader(
        evalset, batch_size=1, shuffle=True)

    dataiter = iter(evalloader)

    outputs = []

    # iterate over all the images
    while True:
        try:
            images, labels = dataiter.next()

            # get output layer -- we assume it has only convolutional layers
            out = model(images)
            out = out.detach().numpy().flatten()
            outputs.append(out)
        except BaseException:
            break

    return outputs


if __name__ == '__main__':

    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir', type=str,
                        help='select input directory for images')
    parser.add_argument('--outputDir', type=str,
                        help='select the output directory for the images')
    options = parser.parse_args()

    # to execute the file standalone
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()

    # extract features of the model
    features = ExtractFeatureLevel(model, options.imgDir)

    ix = 0
    for feat in features:

        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(feat)
        plt.ylabel('some numbers')
        filename = 'graph_' + str(ix) + '.jpg'
        filepath = path.join(options.outputDir, filename)
        plt.savefig(filepath)

        ix += 1
