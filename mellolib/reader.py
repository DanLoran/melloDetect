from PIL import Image
import mellolib.globalConstants
import torch


def readSmallImg(imageName):
    """
        Function to get the tensor of an image that is already correctly sized,
        given its name. Assumes the image has been converted by using dataset/shrink
        thus being with .jpeg extension
    @imageName: a string that describes uniquely an image (ex: ISIC_0034312)
    @returns: a PIL Image
    """
    return Image.open(imageName + "_small.jpeg")


def readImage(imageName):
    """
        Function to get the tensor of an image, correctly resized,
        given its name
    @imageName: a string that describes uniquely an image (ex: ISIC_0034312)
    @returns: a PIL Image
    """
    # attempt to get a .jpeg image
    try:
        img = Image.open(imageName + ".jpeg").resize([256, 256])
        return img
    except FileNotFoundError:
        pass

    # attempt to get a .jpg image if the .jpeg was not found
    try:
        img = Image.open(imageName + ".jpg").resize([256, 256])
        return img
    except FileNotFoundError:
        pass

    # ... else try to get a png image (converted)
    return Image.open(imageName + ".png").convert("RGB").resize([256, 256])


def readVectorImage(imageName, modelName):
    """
        Function to get a vector that corresponds to the features extracted by
        a pre-trained neural network
    @imageName: a string that describes uniquely an image (ex: ISIC_0034312)
    @model: a pre-trained model
    @return: a tensor vector
    """
    filename = imageName + modelName + ".pt"
    map_location = torch.device(
        'gpu' if mellolib.globalConstants.DEPLOY_ON_GPU else 'cpu')
    return torch.load(filename, map_location=map_location)
