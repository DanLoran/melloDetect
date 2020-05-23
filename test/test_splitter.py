import os
from PIL import Image
from mellolib.splitter import Splitter

def create_fake_data(fs, path):
    label_path = path + "label.csv"
    num_neg_images = 6
    num_pos_images = 6

    fs.create_file(label_path)
    with open(label_path, "w") as f:
        for num in range(num_neg_images):
            f.write("ISIC_000000" + str(num) + ",30,0,0\n")
        for num in range(num_neg_images, num_neg_images + num_pos_images):
            f.write("ISIC_000000" + str(num) + ",30,0,1\n")
    
    for num in range(num_neg_images):
        image_path = path + "ISIC_000000" + str(num) + ".jpeg"
        fs.create_file(image_path)
        Image.new('RGB', size=(50, 50), color=(0, 255, 0)).save(image_path)

    for num in range(num_neg_images, num_neg_images + num_pos_images):
        image_path = path + "ISIC_000000" + str(num) + ".jpeg"
        fs.create_file(image_path)
        Image.new('RGB', size=(50, 50), color=(0, 255, 0)).save(image_path)


def test_my_fakefs_test(fs):
    # "fs" is the reference to the fake file system
    # This tests just ensures the fakefs works.
    fs.create_file('/var/data/xx1.txt')
    assert os.path.exists('/var/data/xx1.txt')

def test_basic_splitter(fs):
    data_path = '/test/'

    create_fake_data(fs, data_path)

    splitter = Splitter(data_path, 0.5, 123)

    assert len(splitter.generate_training_data()) == 6
    assert len(splitter.generate_validation_data()) == 6