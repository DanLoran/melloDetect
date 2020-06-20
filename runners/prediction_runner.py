import argparse
import csv
import os
import torch

from mellolib import commonParser as cmp
from mellolib.reader import readImage, readSmallImg, readVectorImage
from mellolib.splitter import Splitter
from torchvision.transforms import ToTensor

def imageLoader(imageName, pretrained_model = None, deploy_on_gpu = False, debug = True):

    if pretrained_model is not None:
        # if a model was specified, get the vector of features
        try:
            image = readVectorImage(
                imageName, pretrained_model, deploy_on_gpu)
            return image
        except Exception as e:
            # print exception in debug mode
            if debug:
                print(e)
            pass

    # try to get small image
    try:
        image = readSmallImg(imageName)
    except FileNotFoundError:
        # ... else try to get normal image
        image = readImage(imageName)

    image = ToTensor()(image).type(torch.float)
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image

### Setup parser
parser = argparse.ArgumentParser()
cmp.prediction_runner(parser)
options = parser.parse_args()

### Choose architecture
model = cmp.model_selection(options.arch)

# Setup GPU
if (options.deploy_on_gpu):
    if (not torch.cuda.is_available()):
        print("GPU device doesn't exist")
    else:
        model = model.cuda()
        print("Deploying model on: " +
            torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")

# validate that the weight path is a file
assert os.path.isfile(options.weight_filepath)

outputFile = open("output.csv", "w")

outputFile.write("image_name,target\n")

with open(os.path.join(options.data_addr, "label.csv"), "r") as f:
    for row in csv.reader(f):

        if options.deploy_on_gpu:
            model.load_state_dict(torch.load(options.weight_filepath))
        else:
            model.load_state_dict(torch.load(options.weight_filepath, map_location=torch.device('cpu')))

        imageName = os.path.join(options.data_addr,row[0])

        model.eval()

        pred = model(imageLoader(imageName,
            pretrained_model=options.pretrained_model,
            debug=True,
            deploy_on_gpu=options.deploy_on_gpu))

        prob = str(pred[0][1].detach().numpy())

        outputFile.write(row[0] + "," + prob + "\n")
