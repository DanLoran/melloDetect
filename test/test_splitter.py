import os
from PIL import Image
from mellolib.splitter import Splitter
from mellolib.augment import white_balancer
from pathlib import Path
from PIL import ImageChops
from shutil import copyfile
import torch

test_file_path = "test/test.jpg"

def create_fake_data(fs):
    path = '/test/'
    Path(path).mkdir(parents=True, exist_ok=True)
    label_path = path + "label.csv"
    num_neg_images = 6
    num_pos_images = 6

    with open(label_path, "w+") as f:
        for num in range(num_neg_images):
            f.write("ISIC_000000" + str(num) + ",30,0,0\n")
        for num in range(num_neg_images, num_neg_images + num_pos_images):
            f.write("ISIC_000000" + str(num) + ",30,0,1\n")
    
    # copy the test image to these locations
    fs.add_real_file(test_file_path)
    for num in range(num_neg_images):
        image_path = path + "ISIC_000000" + str(num) + ".jpeg"
        copyfile(test_file_path, image_path)

    for num in range(num_neg_images, num_neg_images + num_pos_images):
        image_path = path + "ISIC_000000" + str(num) + ".jpeg"
        copyfile(test_file_path, image_path)

    return path

def test_basic_splitter(fs):
    splitter = Splitter(create_fake_data(fs), 0.5, 123)

    assert len(splitter.generate_training_data()) == 6
    assert len(splitter.generate_validation_data()) == 6

    # check we can read the image
    splitter.generate_training_data()[0]


def test_capped_images(fs):
    splitter = Splitter(create_fake_data(fs), 0.5, 123, num_images=2)

    assert len(splitter.generate_training_data()) == 2
    assert len(splitter.generate_validation_data()) == 2

def test_augmented_images(fs):
    splitter = Splitter(create_fake_data(fs), 0.5, 123, num_images=2, augmentations=[white_balancer([1900])])

    assert len(splitter.generate_training_data()) == 4
    assert len(splitter.generate_validation_data()) == 4

    # Check we have actually modified the pixel values
    dataset = splitter.generate_training_data()
    assert not torch.all(torch.eq(dataset[0][0], dataset[1][0]))