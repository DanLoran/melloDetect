import random
import os
import csv
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
from mellolib.globalConstants import FIELDS
from mellolib.augment import identity
from math import floor
from .reader import readSmallImg, readImage, readVectorImage


class SimpleDataset(Dataset):
    '''
    Simply read images, apply augmentations, and return them.
    '''
    # Data is a list of the form [('path', [label]), ...]

    def __init__(self, data, augmentations,
                 pretrained_model=None, deploy_on_gpu=False, debug=False):
        self.data = data
        self.augmentations = augmentations
        self.num_augmentations = sum([a.num for a in augmentations])
        self.pretrained_model = pretrained_model
        self.deploy_on_gpu = deploy_on_gpu
        self.debug = debug

    def __getitem__(self, index):
        base_index = floor(index / self.num_augmentations)
        augmentation_index = index % self.num_augmentations

        # Convert label to torch vector.
        label = torch.FloatTensor(self.data[base_index][1])

        if self.pretrained_model is not None:
            # if a model was specified, get the vector of features
            try:
                inputTensor = readVectorImage(
                    self.data[base_index][0], self.pretrained_model, self.deploy_on_gpu)
                return inputTensor, label
            except Exception as e:
                # print exception in debug mode
                if self.debug:
                    print(e)
                pass

        try:
            # attempt to read small images ...
            image = readSmallImg(self.data[base_index][0])
        except FileNotFoundError:
            # ... and if not found, get normal images
            image = readImage(self.data[base_index][0])

        # pick the augmentation to apply
        for augmentation in self.augmentations:
            if augmentation_index < augmentation.num:
                image = augmentation(image, augmentation_index)
            augmentation_index -= augmentation.num

        # convert from PIL to pytorch tensor
        inputTensor = ToTensor()(image).type(torch.float)

        return inputTensor, label

    def __len__(self):
        return len(self.data) * self.num_augmentations


class Splitter:
    """
    Split datasets into train and validate on the fly.
    Splitter will:
    1) Read all datapoints at labels_path
    2) Shuffle the datapoints
    3) Chop the datapoints so there is an equal number in each class (max num_images)
    4) Return a SimpleDataset which will:
        a) Apply augmentations
        b) Return images
    """

    def __init__(self,
                 labels_path,
                 train_validate_ratio,
                 seed,
                 num_images=None,
                 augmentations=[],
                 pretrained_model=None,
                 deploy_on_gpu=False,
                 debug=False):
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
        num_images : int
            Total number of unique images to read. None means return all.
            (note that the total number of images returned = num_images * # of
            augmentation outputs per image)
        augmentations : [(augmentation_function)]
            Augmentations used to expand the number of images returned. function
            is a function that takes an image and has a "num" property which
            desribes how many images it will return, and yields some number of augmented images.
            e.g. argument: [augment.mirrorer(), white_balancer([1900, 15000])]
        pretrained_model : torch model
            The model is used to obtain a precomputed vector of features instead
            of the image as image input. Default is None.
        deploy_on_gpu: boolean
            Whether to deploy the model on gpu or cpu. Default is false.
        """
        random.seed(seed)
        self.augmentations = augmentations + [identity()]

        self.pretrained_model = pretrained_model
        self.deploy_on_gpu = deploy_on_gpu
        self.debug = debug

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
        indexes = [
            [i for i in range(len(self.data)) if self.data[i]
             [1][FIELDS[label]] == 1]
            for label in FIELDS]

        # We assume we want a 50/50 split on benign/malignant images, so we always trucate to the
        # smallest class. If num_images is set and less than the smallest class, truncate to
        # that instead
        truncation_length = min([len(l) for l in indexes])
        if num_images is not None and num_images < truncation_length:
            truncation_length = num_images

        indexes = [l[:truncation_length] for l in indexes]

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
            self.split_indexes.append(
                [index[split_location:], index[:split_location]])

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
        return SimpleDataset(
            [self.data[i]
                for i in self.split_indexes[0][0] + self.split_indexes[1][0]],
            self.augmentations,
            pretrained_model = self.pretrained_model,
            deploy_on_gpu = self.deploy_on_gpu,
            debug = self.debug)

    def generate_validation_data(self):
        return SimpleDataset(
            [self.data[i]
                for i in self.split_indexes[0][1] + self.split_indexes[1][1]],
            self.augmentations,
            pretrained_model = self.pretrained_model,
            deploy_on_gpu = self.deploy_on_gpu,
            debug = self.debug)
