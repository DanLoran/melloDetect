import random
import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torch
from mellolib.globalConstants import FIELDS
from mellolib.augment import identity
from math import floor

class SimpleDataset(Dataset):
    '''
    Simply read files, apply a transformation (and possibly an augmentation), and return them.
    '''
    # Data is a list of the form [('path', [label]), ...]
    def __init__(self, data, transforms, augmentations):
        self.data = data
        self.transforms = transforms
        self.augmentations = augmentations
        self.num_augmentations = sum([a.num for a in augmentations])

    def __getitem__(self, index):
        # TODO(sam): make this resizing cleaner. Maybe add in pre-transforms?
        image = Image.open(self.data[floor(index / self.num_augmentations)][0]).resize([256, 256])

        # pick the augmentation to apply
        augmentation_index = index % self.num_augmentations
        for augmentation in self.augmentations:
            if augmentation_index < augmentation.num:
                image = augmentation(image, augmentation_index)
            augmentation_index -= augmentation.num

        if self.transforms is not None:
            image = self.transforms(image)

        return image.type(torch.float), torch.FloatTensor(self.data[index][1])

    def __len__(self):
        return len(self.data) * self.num_augmentations

class Splitter:
    """
    Split datasets into train and validate on the fly.
    Splitter will:
    1) Read all datapoints at labels_path
    2) Shuffle the datapoints
    3) Chop the datapoints so there is an equal number in each class
    4) Apply transformations
    5) Apply augmentations
    6) Return images
    """
    def __init__(self,
                 labels_path,
                 train_validate_ratio,
                 seed,
                 transforms=None,
                 num_images=None,
                 augmentations=[]):
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
        transforms : function
            Transformations to apply to the images before they're returned.
        num_images : int
            Total number of unique images to read. None means return all.
            (note that the total number of images returned = num_images * # of augmentation outputs
            per image)
        augmentations : [(function, int)]
            Augmentations used to expand the number of images returned. function is a function that
            takes an image and yields some number of augmented images.
        """
        random.seed(seed)
        self.transforms = transforms
        self.augmentations = augmentations + [identity()]

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
            [i for i in range(len(self.data)) if self.data[i][1][FIELDS[label]] == 1]
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
            self.split_indexes.append([index[split_location:], index[:split_location]])

    def read_data(self, labels_path):
        data = []
        with open(os.path.join(labels_path, "label.csv"), "r") as f:
            for row in csv.reader(f):
                image_path = os.path.join(labels_path, row[0] + ".jpeg")

                # Convert to onehot. If label equals i set 1, else set 0.
                label_value = int(row[3])
                label = [int(label_value == i) for i in range(len(FIELDS))]

                data.append((image_path, label))
        return data

    def generate_training_data(self):
        return SimpleDataset(
            [self.data[i] for i in self.split_indexes[0][0] + self.split_indexes[1][0]],
            self.transforms,
            self.augmentations)

    def generate_validation_data(self):
        return SimpleDataset(
            [self.data[i] for i in self.split_indexes[0][1] + self.split_indexes[1][1]],
            self.transforms,
            self.augmentations)
