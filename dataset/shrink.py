from PIL import Image
import numpy as np
import pydicom as dicom
import os
import csv


def read_shrink_write(path):

    dst = path + "_small.jpeg"

    # .jpeg extension -- need no conversion, just resizing
    if os.path.isfile(path + ".jpeg"):
        Image.open(path + ".jpeg").resize([256, 256]).save(dst)
        return True

    if os.path.isfile(path + ".jpg"):
        Image.open(path + ".jpg").resize([256, 256]).save(dst)
        return True

    # .png extensions -- need conversion to RBG from ARGB, and resizing
    if os.path.isfile(path + ".png"):
        Image.open(path + ".png").convert("RGB").resize([256, 256]).save(dst)
        return True

    # DICOM image format (.dcm)
    if os.path.isfile(path + ".dcm"):
        ds = dicom.read_file(path + ".dcm")
        # store the raw image data
        img = ds.pixel_array

        # use rescale slope and intercept information from the image header
        # and translate it
        PIL_image = Image.fromarray(img).convert("RGB").resize([256, 256])
        PIL_image.save(dst)
        return True

    return False


if __name__ == "__main__":
    with open("label.csv", "r") as f:
        for row in csv.reader(f):
            path = "Data/Images/" + row[0]
            if (not os.path.isfile(path + "_small.jpeg")):

                if read_shrink_write(path):
                    print("Shrunk: " + path)
                else:
                    print("Not found: " + path)
