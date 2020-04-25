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

from visdom import Visdom
from torch.optim import SGD
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
from tqdm import tqdm

from mellolib import CommonParser as cmp
from mellolib.readData import MelloDataSet
from mellolib.globalConstants import ARCH

for library in ARCH:
    try:
        exec("from mellolib.models.{module} import {module}".format(module=library))
    except Exception as e:
        print(e)

# Setup parser
parser = argparse.ArgumentParser()
cmp.basic_runner(parser)
options = parser.parse_args()

# Setup visdom
viz = 0
if(options.show_learning_curve):
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

# transfer learning from resnet18
if options.arch == "zoo-resnet18":
    if (options.deploy_on_gpu):
        model = torch.nn.DataParallel(torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True).cuda())
    else:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
        #TODO: modify the last layer

# example mellolib model
elif options.arch == "tiny-fc":
    model = tiny_fc()

elif options.arch == "tiny-cnn":
    model = tiny_cnn()

elif options.arch == "resnet18":
    model = resnet18()

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

n_eps = 100
optimizer = SGD(model.parameters(), lr=0.001)
criterion = BCELoss()
dataset = MelloDataSet(options.train_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
batch_n = 0
itr = 0
losses = []
time = []

cmp.DEBUGprint("Training... \n", options.debug)

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

        if (options.show_learning_curve):
            losses.append(loss.cpu().detach().numpy())
            time.append(itr)
            viz.line(X=time,Y=losses,win='viz1',name="Learning curve")
            itr+=1
    torch.save(model.state_dict(),options.weight_addr + str(ep))
log.close()

# Begin Validating
if (options.run_validation):
    cmp.DEBUGprint("Validating... \n", options.debug)
    test_dataset = MelloDataSet(options.val_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()
    batch_n = 0
    for inp, target in loader:

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
    print("AUC Results: {}".format(auc))
