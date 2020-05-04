import torch
import os
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from mellolib.globalConstants import FIELDS

class MelloDataSet(Dataset):
    """Class to import csv's containing classification data and features."""

    def __init__(self, labels_path, transforms=None):
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

        with open(os.path.join(labels_path, "label.csv"), "r") as f:
            for row in csv.reader(f):
                self.image_list.append(os.path.join(labels_path, row[0] + ".jpeg"))

                # Convert to onehot. If label equals i set 1, else set 0.
                label = int(row[3])
                self.labels.append([int(label == i) for i in range(len(FIELDS))])

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            Index of the image to be returned.
        output : (<PIL image with a torch.float type>, <torch.FloatTensor>)
        """

        image = Image.open(self.image_list[index])

        if (self.transforms is not None):
            image = self.transforms(image)

        return image.type(torch.float), torch.FloatTensor(self.labels[index])

    def __len__(self):
        return len(self.image_list)
