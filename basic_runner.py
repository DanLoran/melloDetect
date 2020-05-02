################################################################################
# Basic Runner: including validator option
# Assuming Stochastic Gradient Descent optimization method with loss calculated
# with CrossEntropyLoss equation.
# The runner can be executed on the GPU.
# The runner will run for 10 epochs with batch size of 32
# The learning rate is set at 0.0001
# The runner will only consider subset="MALIGNANT" label
# The validation metric is ROC AUC score
# Implementation inspired from UCD Chest X-ray Challenge found at:
# https://github.com/hahnicity/ucd-cxr/blob/master/tutorial/basic.py
# Authors: Gregory Rehm, Minh Truong
################################################################################

import torch
import argparse
import numpy as np

from visdom import Visdom
from torch.optim import SGD, Adam
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
from tqdm import tqdm
from datetime import datetime

from mellolib import commonParser as cmp
from mellolib.readData import MelloDataSet
from mellolib.globalConstants import ARCH
from mellolib.models import *

"""
Evaluation function: validates data
@returns: (float) Area under the curve - auc
"""
def run_evaluation(test_loader):
    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()
    batch_n = 0
    for inp, target in test_loader:

        if (options.deploy_on_gpu):
            target = torch.autograd.Variable(target).cuda()
            inp = torch.autograd.Variable(inp).cuda()
        else:
            target = torch.autograd.Variable(target)
            inp = torch.autograd.Variable(inp)

        out = model(inp)
        # Add results of the model's output to the aggregated prediction vector, and also add aggregated
        # ground truth information as well
        pred = torch.cat((pred, out.cpu().detach()), 0)
        gt = torch.cat((gt, target.cpu().detach()), 0)

    # Compute the model area under curve (AUC).
    auc = roc_auc_score(gt, pred)
    return auc

for library in ARCH:
    try:
        if ('trans' not in library):
            exec("from mellolib.models.{module} import {module}".format(module=library))
    except Exception as e:
        print(e)

# Setup parser
parser = argparse.ArgumentParser()
cmp.basic_runner(parser)
options = parser.parse_args()

# Setup visdom
viz = 0
if(options.show_visdom):
    try:
        viz = Visdom()
    except Exception as e:
        print(
            "The visdom experienced an exception while running: {}\n"
            "The demo displays up-to-date functionality with the GitHub "
            "version, which may not yet be pushed to pip. Please upgrade "
            "using `pip install -e .` or `easy_install .`\n"
            "If this does not resolve the problem, please open an issue on "
            "our GitHub.".format(repr(e))
        )

# Choose architecture
cmp.DEBUGprint("Loading model. \n", options.debug)
model = 0

# example mellolib model
if options.arch == "tiny_fc":
    model = tiny_fc()

elif options.arch == "tiny_cnn":
    model = tiny_cnn()

elif options.arch == "lorenzo_resnet18":
    model = lorenzo.resnet18()

elif options.arch == "trans_resnet18":
    model = transfer.resnet18()

elif options.arch == "trans_mobilenet":
    model = transfer.mobilenet()

elif options.arch == "trans_alexnet":
    model = transfer.alexnet()

elif options.arch == "trans_vgg":
    model = transfer.vgg()

elif options.arch == "trans_densenet":
    model = transfer.densenet()

elif options.arch == "trans_inception":
    model = transfer.inception()

elif options.arch == "trans_googlenet":
    model = transfer.googlenet()

elif options.arch == "trans_shufflenet":
    model = transfer.shufflenet()

else:
    print("Architecture don't exist!")
    exit(1)

# If deploy on gpu
if (options.deploy_on_gpu):
    if (not torch.cuda.is_available()):
        print("GPU device doesn't exist")
    else:
        model = model.cuda()
        print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")

# If resume training
cmp.DEBUGprint("Loading previous state. \n", options.debug)
if options.run_at_checkpoint:
    model.load_state_dict(torch.load(options.weight_addr))

# Setup log
cmp.DEBUGprint("Setup log. \n", options.debug)
log = open(options.log_addr,"w+")

# Basic runner stuff
cmp.DEBUGprint("Initialize runner. \n", options.debug)

# TODO: move the capitalized variables to configuration file or elsewhere
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = BCELoss()
dataset = MelloDataSet(options.train_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
batch_n = 0
itr = 0
losses = []
time = []

# evaluation parameters
if (options.run_validation):
    test_dataset = MelloDataSet(options.val_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    eval_score = []

cmp.DEBUGprint("Training... \n", options.debug)

# current date and time
now = datetime.now()

date = datetime.timestamp(now)
timestamp = datetime.fromtimestamp(date)
print("Start training at ", timestamp)

# Begin Training (ignore tqdm, it is just a progress bar GUI)
model.train()
for ep in tqdm(range(EPOCHS)):
    for inp, target in loader:
        if options.deploy_on_gpu:
            target = torch.autograd.Variable(target).cuda()
            inp = torch.autograd.Variable(inp).cuda()
        else:
            target = torch.autograd.Variable(target)
            inp = torch.autograd.Variable(inp)

        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        log.write(str(loss.cpu().detach().numpy().item()))
        log.write("\n")

        if (options.show_visdom):
            losses.append(loss.cpu().detach().numpy())
            time.append(itr)
            viz.line(X=time,Y=losses,win='viz1',name="Learning curve", opts={'linecolor': np.array([[0, 0, 255],]), 'title':"Learning curve"})
            itr+=1

        if options.run_validation:
            # evaluate the model
            eval_score.append(run_evaluation(test_loader))
            if options.show_visdom:
                viz.line(X =time, Y = eval_score, win='viz2', name="Evaluation AUC",  opts={'linecolor': np.array([[255, 0, 0],]), 'title':"AUC score"})
            else:
                print("AUC: %f" %(eval_score[-1]))
    if options.checkpoint:
        torch.save(model.state_dict(),options.weight_addr + str(timestamp) + "_epoch_" +  str(ep))
log.close()
