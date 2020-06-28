from mellolib.models.pretrained import  validatePretrained, getPretrainedModelNoFc
from mellolib.reader import readImage, readSmallImg
import csv
import argparse
import sys
import os
from torchvision.transforms import ToTensor
import torch
sys.path.append('/w')

def imageLoader(imageName):
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

def featurize(imageName, model):

    features = model(imageLoader(imageName))
    return features

if __name__ == "__main__":

    # parse model name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="the pretrained model to use",
        type=str,
        default="resnet34")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="the directory where the images and label are located"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default = "", help="output directory"
    )
    args = parser.parse_args()

    # validate the model name
    validatePretrained(args.model)

    # define output directory if not specified
    if args.output == "":
        args.output = args.data_dir

    # get the model
    model = getPretrainedModelNoFc(args.model)
    model.eval()

    with open(os.path.join(args.data_dir, "label.csv"), "r") as f:
        for row in csv.reader(f):

            # build imagename and filename
            imageName = os.path.join(args.data_dir,row[0])
            filename = os.path.join(args.output, row[0]) + args.model + ".pt"

            if (not os.path.isfile(filename)):
                # extract and save features
                try:
                    features = featurize(imageName, model)
                    torch.save(features, filename)
                    print("Featurized: " + imageName)
                except:
                    print("Not found:" + imageName)
