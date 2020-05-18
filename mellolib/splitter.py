import numpy as np
import random
import os
import csv
from PIL import Image
from mellolib.globalConstants import FIELDS
from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    # Data is a list of the form [('path', [label]), ...]
    def __init__(self, data, transforms, opt, gpu):
        self.data = data
        self.transforms = transforms
        self.opt = opt
        self.gpu = gpu

        # Preprocessing before optimization level 1 or 2
        path = self.data[0][0].split('/')
        path = '/'.join(path[0:-1]) + '/'

        if (self.transforms is None and self.opt > 0):
            print("Error: Optimization level > 0 needs pre-training transformation")
            exit(-1)

        if (self.opt == 1):
            for index in range(len(self.data)):
                image = Image.open(self.data[index][0])
                image = self.transforms(image)
                torch.save(image.type(torch.float), path + str(index) + '.pt')
                self.data[index] = (path + str(index) + '.pt', self.data[index][1])

    def __getitem__(self, index):

        if (self.opt == 0):
            image = Image.open(self.data[index][0])
        elif (self.opt == 1 and self.gpu):
            image = torch.load(self.data[index][0], map_location=lambda storage, loc: storage.cuda(0))
        elif (self.opt == 1 and not self.gpu):
            image = torch.load(self.data[index][0])


        if (self.transforms is not None and self.opt == 0):
            image = self.transforms(image)
            image = image.type(torch.float)

        return image, torch.FloatTensor(self.data[index][1])

    def __len__(self):
        return len(self.data)

class Splitter:
    """Split datasets into train and validate on the fly"""
    def __init__(self, labels_path, train_validate_ratio, seed, transforms=None, opt=0, gpu=True):
        """
        Parameters
        ----------
        labels_path : String
            Location of labels.csv
        train_validate_ratio : float
            The ratio between the number of training and validation samples.
            0 means no training data, 1 means only training data
        seed : int
            Seed to use for random.seed. Don't change without good reason.

        opt : int (0-2)
            Optimization level that effects the training time.
            0: read with PIL (simplest, most straight-forward way to load in images)
            1: read with Pytorch Tensor pickle load (speedup ammortized over # epochs)
            2: read with HDF5 database
        """
        self.gpu = gpu
        self.opt = opt

        random.seed(seed)
        self.transforms = transforms

        # self.data contains a list of the form [('path', [label onehot]), ...].
        # This is the base data that all indexes below refer to.
        self.data = self.read_data(labels_path)
        random.shuffle(self.data)

        # 0: benign, 1: malignant.
        # Indexes is a 2d array of the form
        # [
        #   [benign label indexes],
        #   [malignant label indexes]
        # ]
        indexes = [[i for i in range(len(self.data)) if self.data[i][1][FIELDS[label]] == 1] for label in FIELDS]

        # split_indexes is a 3d array of the form:
        # [
        #   [
        #     [benign training images],
        #     [benign validation images]
        #   ],
        #   [
        #     [malignant training images],
        #     [malignant validation images]
        #   ]
        # ]
        self.split_indexes = []
        for index in indexes:
            split_location = int(len(index) * train_validate_ratio)
            self.split_indexes.append([index[split_location:], index[:split_location]])

    def read_data(self, labels_path):
        data = []
        with open(os.path.join(labels_path, "label.csv"), "r") as f:
            for row in csv.reader(f):
                image_path = os.path.join(labels_path, row[0])

                # Convert to onehot. If label equals i set 1, else set 0.
                label_value = int(row[3])
                label = [int(label_value == i) for i in range(len(FIELDS))]

                data.append((image_path, label))
        return data

    def generate_training_data(self):
        return SimpleDataset([self.data[i] for i in self.split_indexes[0][0] + self.split_indexes[1][0]], self.transforms, self.opt, self.gpu)

    def generate_validation_data(self):
        return SimpleDataset([self.data[i] for i in self.split_indexes[0][1] + self.split_indexes[1][1]], self.transforms, self.opt, self.gpu)
