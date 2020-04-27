import torch
import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from mellolib.globalConstants import FIELDS

class MelloDataSet(Dataset):
    def __init__(self, data_dir, subset=None, transforms=None):
        image_list = []
        labels = []

        if (subset is not None):
            subset_idx = FIELDS[subset]

        with open(data_dir+"label.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                image_name = row[0] + '.jpeg'

                if (subset is not None):
                    label = [int(row[3:][subset_idx])]

                else:
                    label = row[3:]
                    label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_list.append(image_name)
                labels.append(label)

        self.image_list = image_list
        self.labels = labels
        self.transforms = transforms
        self.safe_image = 0
        self.safe_label = 0

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.labels[index]

        try:
            image = Image.open(image_name)
        except FileNotFoundError:
            image = self.safe_image
            label = self.safe_label
            return image.type(torch.float), torch.FloatTensor(label)

        if (self.transforms is not None):
            image = self.transforms(image)

        self.safe_image = image
        self.safe_label = label

        return image.type(torch.float), torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_list)
