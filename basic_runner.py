################################################################################
# Basic Runner: including validator option
# Assuming Stochastic Gradient Descent optimization method with loss calculated
# with BCE equation.
# The runner can be executed on the GPU.
# The runner will run for 10 epochs with batch size of 32
# The learning rate is set at 0.01
# The validation metric is ROC AUC score
# Implementation inspired from UCD Chest X-ray Challenge found at:
# https://github.com/hahnicity/ucd-cxr/blob/master/tutorial/basic.py
# Authors: Gregory Rehm, Minh Truong
################################################################################

import torch
import argparse
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.optim import SGD
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score

from mellolib import CommonParser
from mellolib.readData import MelloDataSet

# Setup parser
parser = argparse.ArgumentParser()
CommonParser.basic_runner(parser)
options = parser.parse_args()

# Choose architecture
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
model = models.resnet18
if options.arch == "zoo-resnet18":
    model = models.resnet18
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
else:
    print("Architecture don't exist!")
    exit(1)

# If resume training
if options.run_at_checkpoint:
    model.load_state_dict(options.weight_addr)

# Setup log
log = open(options.log_addr,"w")

# Make sure we can run the model on GPU
model = torch.nn.DataParallel(model)
model.train()

# Basic runner stuff
cur = -1
n_eps = 10 - cur - 1
optimizer = SGD(model.parameters(), lr=0.001)
criterion = BCELoss()
dataset = MelloDataSet(options.train_addr)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
batch_n = 0

# Begin Training
for ep in range(n_eps):
    for inp, target in loader:

        if options.deploy_on_gpu:
            target = torch.autograd.Variable(target).cuda()
            inp = torch.autograd.Variable(inp).cuda()

        out = model(inp)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        log.write(str(loss.cpu().detach().numpy().item()))
        log.write("\n")

    torch.save(model.state_dict(),options.weight_addr + str(ep+cur+1))

# Begin Validating
if (options.run_validation):
    test_dataset = MelloDataSet(options.val_addr)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    if (options.deploy_on_gpu):
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()


    model.eval()
    batch_n = 0
    for inp, target in loader:
        if (options.deploy_on_gpu):
            target = torch.autograd.Variable(target).cuda()
            inp = torch.autograd.Variable(inp).cuda()
        out = model(inp)
        # Add results of the model's output to the aggregated prediction vector, and also add aggregated
        # ground truth information as well
        pred = torch.cat((pred, out.data), 0)
        gt = torch.cat((gt, target.data), 0)
        print("end batch")

    # Compute the model area under curve (AUC).
    auc = compute_AUCs(gt, pred)
    print("AUC Results: {}".format(sum(auc) / len(auc)))
