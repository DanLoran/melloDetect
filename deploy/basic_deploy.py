import io
import json

import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import torch
import argparse

import sys
sys.path.append('../')

from mellolib.models import *

############################### Flask Setup ####################################
app = Flask(__name__)

############################### Model Setup ####################################
model = transfer.resnet18()
model.load_state_dict(torch.load('../weight/epoch4'))
model.eval()
############################## Inference Setup #################################

def transform_image(img_addr):
    my_transforms = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
    img = Image.open(img_addr)
    return my_transforms(img)[None, :, :, :]


def get_prediction(img):
    inp = transform_image(img)
    out = model(inp)
    benign = str(out[0][0].item())
    malignant = str(out[0][1].item())
    return benign, malignant

############################# REST setup #######################################
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_addr = request.data.decode("utf-8")
        benign, malignant = get_prediction(img_addr)
    return jsonify({'Benign confidence': benign, 'Malignant cofidence': malignant})

############################## Run #############################################
if __name__ == '__main__':
    app.run()
