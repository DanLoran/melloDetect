from PIL import Image
import os
import csv

def read_shrink_write(path):
    if os.path.isfile(path + ".jpeg"):
        Image.open(path + ".jpeg").resize([256,256]).save(path + "_small.jpeg")
    if os.path.isfile(path + ".png"):
        Image.open(path + ".png").convert("RGB").resize([256,256]).save(path + "_small.jpeg")

if __name__ == "__main__":
    with open("Data/Images/label.csv", "r") as f:
        for row in csv.reader(f):
            path = "Data/Images/" + row[0]
            if (not os.path.isfile(path + "_small.jpeg")):
                read_shrink_write(path)
                print("Shrunk: " + path)