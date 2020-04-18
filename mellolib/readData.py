import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from mellolib.globalConstants import FIELDS

class MelloDataSet(Dataset):
    def __init__(self, data_dir, subset=None):
        image_list = []
        labels = []

        if (subset is not None):
            subset_idx = FIELDS.index(subset)

        with open(data_dir+"label.txt", "r") as f:
            for line in f:
                entry = line.split()
                image_name = entry[0]

                if (subset is not None):
                    label = [int(entry[1:][subset_idx])]

                else:
                    label = entry[1:]
                    label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_list.append(image_name)
                labels.append(label)

        self.image_list = image_list
        self.labels = labels

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.labels[index]
        image = Image.open(image_name)
        image = torch.from_numpy(np.asarray(image))
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_list)
