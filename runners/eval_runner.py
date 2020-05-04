import torch
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('../')

from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale

from mellolib import commonParser as cmp
from mellolib.readData import MelloDataSet
from mellolib.globalConstants import ARCH
from mellolib.models import transfer
from mellolib.eval import eval_auc, eval_accuracy, eval_f1, eval_precision, eval_recall, generate_results

############################ Setup parser ######################################
parser = argparse.ArgumentParser()
cmp.eval_runner(parser)
options = parser.parse_args()

########################## Choose architecture #################################
model = cmp.model_selection(options.arch)

test_dataset = MelloDataSet(options.val_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
eval_score = []

########################### Environment setup ##################################
# Setup GPU
if (options.deploy_on_gpu):
    if (not torch.cuda.is_available()):
        print("GPU device doesn't exist")
    else:
        model = model.cuda()
        print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")

############################### Evaluate #######################################
weight_list = [f for f in listdir(options.eval_weight_addr) if isfile(join(options.eval_weight_addr, f))]
weight_list.sort()
eval_score = []

for weight_addr in weight_list:

    print("+"*60)
    print("For " + weight_addr)
    print("+"*60)

    model.load_state_dict(torch.load(options.eval_weight_addr + weight_addr))
    gt, pred = generate_results(test_loader, options, model)

    auc = eval_auc(gt,pred)
    acc = eval_accuracy(gt, pred)
    f1  = eval_f1(gt,pred)
    pre = eval_precision(gt,pred)
    rec = eval_recall(gt,pred)

    print("AUC: " + str(auc))
    print("ACC: " + str(acc))
    print("F1: "  + str(f1))
    print("PRE: " + str(pre))
    print("REC: " + str(rec))
