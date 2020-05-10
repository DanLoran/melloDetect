import torch
import math
import os
import re
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from mellolib.globalConstants import FIELDS

"""
dataPath (string): the path to the directory that contains the data. It assumes
    it has a subdirectory "Images" with all the images, and a subdirectory
    "Descriptions" with all labels.
trainingRatio (float): the ratio [0,1] of images that are part of the training
    set
"""
def Split(dataPath, trainingRatio, transforms = None):
    ImagePath = os.path.join(dataPath, "Images")
    imageNameArray = np.array(sorted(os.listdir(ImagePath)))

    labelPath = os.path.join(dataPath, "Descriptions")
    labelNameArray = np.array(sorted(os.listdir(labelPath)))

    # check that both images and labels have the same size
    assert labelNameArray.shape[0] == imageNameArray.shape[0]
    size = labelNameArray.shape[0]

    # create a permutation of both labels and images
    p = np.random.permutation(size)

    # shuffle the data
    imageNameArray = imageNameArray[p]
    labelNameArray = labelNameArray[p]

    trainingSize = math.floor(trainingRatio * size)

    trainImageNames = imageNameArray[:trainingSize]
    testImageNames = imageNameArray[trainingSize:]
    trainLabelNames = labelNameArray[:trainingSize]
    testLabelNames = labelNameArray[trainingSize:]

    trainDataSet = MelloDataSet(dataPath, trainImageNames, trainLabelNames, transforms)
    testDataSet = MelloDataSet(dataPath, testImageNames, testLabelNames, transforms)

    return trainDataSet, testDataSet


class MelloDataSet(Dataset):
    """Class to import csv's containing classification data and features."""

    def __init__(self, path, imageNamesArray, labelNamesArray, transforms=None):
        """
        Parameters
        ----------
        data_dir : str
            Contains the absolute base filepath for the labels csv.
        transforms : function
            transformations to be applied to the image when read.
        """

        self.labels = []
        self.image_list = []
        self.transforms = transforms
        self.path = path

        for i, file in enumerate(labelNamesArray.tolist()):
            f = open(os.path.join(path,"Descriptions",file))
            filetext = f.readlines()

            targetstr = 'benign_malignant'
            res = [x for x in filetext if re.search(targetstr, x)]

            if res == []:
                continue

            if('"benign"' in res[0]):
                self.labels.append([1,0])
            elif('"malignant"' in res[0]):
                self.labels.append([0,1])
            else:
                continue

            self.image_list.append(imageNamesArray[i])

        assert len(self.image_list) == len(self.labels)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            Index of the image to be returned.
        output : (<PIL image with a torch.float type>, <torch.FloatTensor>)
        """

        image = Image.open(os.path.join(self.path, "Images", self.image_list[index]))

        if (self.transforms is not None):
            image = self.transforms(image)

        return image.type(torch.float), torch.FloatTensor(self.labels[index])

    def __len__(self):
        return len(self.image_list)
