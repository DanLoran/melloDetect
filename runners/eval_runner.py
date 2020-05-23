import torch
import argparse
import numpy as np
from os import listdir, path
from os.path import isfile, join
import sys
sys.path.append('../')

from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale

from mellolib.splitter import Splitter
from mellolib import commonParser as cmp
from mellolib.globalConstants import ARCH
from mellolib.models import transfer
from mellolib.eval import eval_auc, eval_accuracy, eval_f1, eval_precision, eval_recall, eval_confuse, generate_results

############################ Setup parser ######################################
parser = argparse.ArgumentParser()
cmp.eval_runner(parser)
options = parser.parse_args()

########################## Choose architecture #################################
model = cmp.model_selection(options.arch)

dataset_generator = Splitter(options.data_addr, options.split, options.seed, transforms=Compose([Resize((256,256)), ToTensor()]))
test_dataset = dataset_generator.generate_validation_data()
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

    weightFileName = path.join(options.eval_weight_addr, weight_addr)

    if options.deploy_on_gpu:
        model.load_state_dict(torch.load(weightFileName))
    else:
        model.load_state_dict(torch.load(weightFileName, map_location=torch.device('cpu')))
    gt, pred = generate_results(test_loader, options, model)

    auc = eval_auc(gt,pred)
    acc = eval_accuracy(gt, pred)
    f1  = eval_f1(gt,pred)
    pre = eval_precision(gt,pred)
    rec = eval_recall(gt,pred)
    tn, tp, fn, fp = eval_confuse(gt,pred)

    print("AUC: " + str(auc))
    print("ACC: " + str(acc))
    print("F1: "  + str(f1))
    print("PRE: " + str(pre))
    print("REC: " + str(rec))
    print("TN: " + str(tn))
    print("TP: " + str(tp))
    print("FN: " + str(fn))
    print("FP: " + str(fp))
