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

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    return outputs

############################# REST setup #######################################
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        benign, malignant = get_prediction(image_bytes=img_bytes)
        return jsonify({'Benign confidence': benign, 'Malignant cofidence': malignant})

############################## Run #############################################
if __name__ == '__main__':
    app.run()
