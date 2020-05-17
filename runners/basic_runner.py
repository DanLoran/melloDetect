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
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('../')

from mellolib import commonParser as cmp
from mellolib.splitter import Splitter
from mellolib.globalConstants import ARCH
from mellolib.models import *
from mellolib.eval import eval_auc, generate_results

############################ Setup parser ######################################
parser = argparse.ArgumentParser()
cmp.basic_runner(parser)
options = parser.parse_args()

##################### Setup visdom visualization ###############################
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

########################## Choose architecture #################################
cmp.DEBUGprint("Loading model. \n", options.debug)
model = cmp.model_selection(options.arch)

########################### Environment setup ##################################

# Setup GPU
if (options.deploy_on_gpu):
    if (not torch.cuda.is_available()):
        print("GPU device doesn't exist")
    else:
        model = model.cuda()
        print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")

# Resume from checkpoint option
cmp.DEBUGprint("Loading previous state. \n", options.debug)
if options.run_at_checkpoint:
    model.load_state_dict(torch.load(options.weight_addr))

# Setup log
cmp.DEBUGprint("Setup log. \n", options.debug)
log = open(options.log_addr,"w+")

# Basic runner stuff
cmp.DEBUGprint("Initialize runner. \n", options.debug)

########################## Training setup ######################################
n_eps = 10
batch_size = 32
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)
criterion = BCELoss()
dataset_generator = Splitter(options.data_addr, options.split, options.seed)
dataset = dataset_generator.generate_training_data()
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
batch_n = 0
itr = 0
losses = []
time = []

# evaluation parameters
if (options.run_validation):
    test_dataset = dataset_generator.generate_validation_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    eval_score = []

cmp.DEBUGprint("Training... \n", options.debug)


############################# Training loop ####################################
# current date and time
now = datetime.now()

date = datetime.timestamp(now)
timestamp = datetime.fromtimestamp(date)
print("Start training at ", timestamp)

# Begin Training (ignore tqdm, it is just a progress bar GUI)
model.train()
for ep in tqdm(range(n_eps)):
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

        if options.show_visdom:
            losses.append(loss.cpu().detach().numpy())
            time.append(itr)
            viz.line(X=time,Y=losses,win='viz1', name="Learning curve",
            opts={'linecolor': np.array([[0, 0, 255],]), 'title':"Learning curve"})
            itr+=1

        if options.run_validation:
            gt, pred = generate_results(test_loader, options, model)
            eval_score.append(eval_auc(gt, pred))
            if options.show_visdom:
                viz.line(X =time, Y = eval_score, win='viz2', name="Evaluation AUC",
                opts={'linecolor': np.array([[255, 0, 0],]), 'title':"AUC score"})
            else:
                print("AUC score:" + str(eval_score))
    if options.checkpoint:
        if (options.run_at_checkpoint):
            dir = options.weight_addr.split('/')
            save_name = ''
            for i in range(len(dir) - 1):
                save_name += '/'
                save_name += dir[i]
        else:
            save_name = options.weight_addr
        torch.save(model.state_dict(),save_name + str(timestamp) + "_epoch" +  str(ep))
log.close()
