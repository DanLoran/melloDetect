import numpy as np
import random
import os
import csv
from PIL import Image
from mellolib.globalConstants import FIELDS
from torch.utils.data import Dataset
import torch
import h5py

class L0Dataset(Dataset):
    # Data is a list of the form [('path', [label]), ...]
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.data[index][0])

        if (self.transforms is not None):
            image = self.transforms(image)

        return image.type(torch.float), torch.FloatTensor(self.data[index][1])

    def __len__(self):
        return len(self.data)

class L1Dataset(Dataset):
    # Data is a list of the form [('path', [label]), ...]
    def __init__(self, data, transforms, gpu):
        self.data = data
        self.transforms = transforms
        self.gpu = gpu

        # Preprocessing before optimization level 1 or 2
        path = self.data[0][0].split('/')
        path = '/'.join(path[0:-1]) + '/'

        if (self.transforms is None):
            print("Error: Optimization level > 0 needs pre-training transformation")
            exit(-1)

        for index in range(len(self.data)):
            image = Image.open(self.data[index][0])
            image = self.transforms(image)
            torch.save(image.type(torch.float), path + str(index) + '.pt')
            self.data[index] = (path + str(index) + '.pt', self.data[index][1])

    def __getitem__(self, index):

        if (self.gpu):
            image = torch.load(self.data[index][0], map_location=lambda storage, loc: storage.cuda(0))
        else:
            image = torch.load(self.data[index][0])

        return image, torch.FloatTensor(self.data[index][1])

    def __len__(self):
        return len(self.data)

class L2Dataset(Dataset):
    # Data is a list of the form [('path', [label]), ...]
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
        path = self.data[0][0].split('/')
        path = '/'.join(path[0:-1]) + '/'
        os.system('rm -rf ' + path+'/database.h5')
        self.database = h5py.File(path+'/database.h5','a')
        self.prefetch_image = self.database.create_group('image')
        self.prefetch_label = self.database.create_group('label')

        self.cur_images = []
        self.cur_labels = []

        # These are just heuristic
        self.prefetch_head = 0
        self.max_miss_percentage = 0.01
        self.miss_count = 0
        self.prefetch_length = 256

        if (self.transforms is None):
            print("Error: Optimization level > 0 needs pre-training transformation")
            exit(-1)

        count = 0
        array_name = 0
        for index in range(len(self.data) - (len(self.data) % self.prefetch_length) + 1):
            if(count == self.prefetch_length):
                print("Creating prefetch array " + str(array_name))
                self.prefetch_image.create_dataset(str(array_name), data=self.cur_images,compression="gzip")
                self.prefetch_label.create_dataset(str(array_name), data=self.cur_labels,compression="gzip")
                array_name += 1
                count = 0
                self.cur_images = []
                self.cur_labels = []

            image = Image.open(self.data[index][0])
            image = np.asarray(self.transforms(image), dtype=np.int8)
            label = np.asarray(self.data[index][1], dtype=np.int8)
            self.cur_images.append(image)
            self.cur_labels.append(label)
            count += 1

        self.cur_images = torch.from_numpy(np.asarray(self.prefetch_image.get('0'))).type(torch.float)
        self.cur_labels = torch.from_numpy(np.asarray(self.prefetch_label.get('0'))).type(torch.float)

    def __getitem__(self, index):

        if (index >= len(self.data) - (len(self.data) % self.prefetch_length)):
            return self.cur_images[0],  self.cur_labels[0]

        local_idx = index % self.prefetch_length
        # Read hit
        if (self.prefetch_head <= index and index < self.prefetch_head + self.prefetch_length):
            image = self.cur_images[local_idx]
            label = self.cur_labels[local_idx]

        # Read miss
        else:
            # Tolerate miss
            if (self.miss_count < int(self.prefetch_length * self.max_miss_percentage)):
                self.miss_count += 1
                array_name = int(index/self.prefetch_length)
                temp_image = torch.from_numpy(np.asarray(self.prefetch_image.get(str(array_name)))).type(torch.float)
                temp_label = torch.from_numpy(np.asarray(self.prefetch_label.get(str(array_name)))).type(torch.float)
                image = temp_image[local_idx]
                label = temp_label[local_idx]
            # Reload miss
            else:
                self.miss_count = 0
                array_name = int(index/self.prefetch_length)
                self.cur_images = torch.from_numpy(np.asarray(self.prefetch_image.get(str(array_name)))).type(torch.float)
                self.cur_labels = torch.from_numpy(np.asarray(self.prefetch_label.get(str(array_name)))).type(torch.float)
                self.prefetch_head = array_name * self.prefetch_length
                image = self.cur_images[local_idx]
                label = self.cur_labels[local_idx]

        return image, label

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
                image_path = os.path.join(labels_path, row[0] + ".jpeg")

                # Convert to onehot. If label equals i set 1, else set 0.
                label_value = int(row[3])
                label = [int(label_value == i) for i in range(len(FIELDS))]

                data.append((image_path, label))
        return data

    def generate_training_data(self):
        if (self.opt == 0):
            return L0Dataset([self.data[i] for i in self.split_indexes[0][0] + self.split_indexes[1][0]], self.transforms)
        elif (self.opt == 1):
            return L1Dataset([self.data[i] for i in self.split_indexes[0][0] + self.split_indexes[1][0]], self.transforms, self.gpu)
        elif (self.opt == 2):
            return L2Dataset([self.data[i] for i in self.split_indexes[0][0] + self.split_indexes[1][0]], self.transforms)

    def generate_validation_data(self):
        if (self.opt == 0):
            return L0Dataset([self.data[i] for i in self.split_indexes[0][1] + self.split_indexes[1][1]], self.transforms)
        elif (self.opt == 1):
            return L1Dataset([self.data[i] for i in self.split_indexes[0][1] + self.split_indexes[1][1]], self.transforms, self.gpu)
        elif (self.opt == 0):
            return L2Dataset([self.data[i] for i in self.split_indexes[0][1] + self.split_indexes[1][1]], self.transforms)
