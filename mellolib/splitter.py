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
from math import isclose


class SimpleDataset(Dataset):
    '''
    Simply read images, apply augmentations, and return them.
    '''
    # Data is a list of the form [('path', [label]), ...]

    def __init__(self, data, augmentations, pretrained_model=None,
                 debug=False, use_sex=False):
        self.data = data
        self.augmentations = augmentations
        self.num_augmentations = sum([a.num for a in augmentations])
        self.pretrained_model = pretrained_model
        self.debug = debug
        self.use_sex = use_sex

    def __getitem__(self, index):
        base_index = floor(index / self.num_augmentations)
        augmentation_index = index % self.num_augmentations

        # Convert label to torch vector.
        label = torch.FloatTensor(self.data[base_index][1])

        if self.pretrained_model is not None:
            # if a model was specified, get the vector of features
            try:
                inputTensor = readVectorImage(
                    self.data[base_index][0], self.pretrained_model).flatten()

                if self.use_sex:
                    # concatenate a torch tensor containing the sex information
                    sex = torch.FloatTensor([self.data[base_index][2]])
                    inputTensor = torch.cat((inputTensor, sex), 0)

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
                 debug=False,
                 use_sex=False,
                 positive_case_percent = 0.5):
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
        debug: bool
            Whether to print debug information. Default is False
        use_sex: bool
            Whether to use sex as additional feature to append to the features
            vector. pretrained_model must not be None. Default is False
        positive_case_percent: float
            percentage of the number of cases that are positive (malignant).
            0 -> all negative
            1 -> all positive
            .5 -> 50/50 split
            .75 -> 75% positive
            etc.
        """
        random.seed(seed)
        self.augmentations = augmentations + [identity()]
        self.pretrained_model = pretrained_model
        self.debug = debug
        self.use_sex = use_sex
        if use_sex:
            assert self.pretrained_model != None

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

        # If we have fewer positive examples than we want, truncate the
        # negative cases. Else truncate the positive cases.
        # example: 10 pos, 100 neg and positive_case_percent = .25
        # data_case_percent = .1
        # to_truncate_index = 0
        # final_truncated_size = 10 * ((1 / .25) - 1) = 10 * 3 = 30
        #
        # so we truncate benign to 30, leaving us with 10 pos and 30 neg
        num_pos_cases = len(indexes[0])
        num_total_cases = len(indexes[0]) + len(indexes[1])
        if num_pos_cases / num_total_cases < positive_case_percent:
            final_truncated_size = floor(num_pos_cases * ((1 / positive_case_percent) - 1))
            indexes[0] = indexes[0][:final_truncated_size]
        else:
            final_truncated_size = floor(positive_case_percent * num_total_cases)
            indexes[1] = indexes[1][:final_truncated_size]

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
                [index[:split_location], index[split_location:]])

    def read_data(self, labels_path):
        data = []
        with open(os.path.join(labels_path, "label.csv"), "r") as f:
            for row in csv.reader(f):
                image_path = os.path.join(labels_path, row[0])

                # Convert to onehot. If label equals i set 1, else set 0.
                label_value = int(row[3])
                label = [int(label_value == i) for i in range(len(FIELDS))]

                # read sex from current row
                sex = int(row[2])
                data.append((image_path, label, sex))
        return data

    def generate_training_data(self):
        return SimpleDataset(
            [self.data[i]
                for i in self.split_indexes[0][0] + self.split_indexes[1][0]],
            self.augmentations,
            pretrained_model=self.pretrained_model,
            debug=self.debug,
            use_sex=self.use_sex)

    def generate_validation_data(self):
        return SimpleDataset(
            [self.data[i]
                for i in self.split_indexes[0][1] + self.split_indexes[1][1]],
            self.augmentations,
            pretrained_model=self.pretrained_model,
            debug=self.debug,
            use_sex=self.use_sex)
