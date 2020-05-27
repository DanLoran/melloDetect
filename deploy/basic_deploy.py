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
model.load_state_dict(torch.load('../weight/'))
model.eval()
############################## Inference Setup #################################

def transform_image(img):
    my_transforms = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
    img = Image.open(io.BytesIO(img))
    return my_transforms(img)


def get_prediction(img):
    tensor = transform_image(img)
    outputs = model.forward(tensor)
    return outputs

############################# REST setup #######################################
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_addr = request.files['file']
        img = Image.open(img_addr)
        benign, malignant = get_prediction(img)
        return jsonify({'Benign confidence': benign, 'Malignant cofidence': malignant})

############################## Run #############################################
if __name__ == '__main__':
    app.run()
