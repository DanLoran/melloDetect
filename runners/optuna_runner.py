import torch
import argparse
import numpy as np

import torch.optim as optim
from torch.nn import BCELoss
import torch.nn as nn
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
import torch.nn.functional as F
from collections import OrderedDict

import sys
sys.path.append('../')

from mellolib.eval import generate_results, eval_accuracy
from mellolib import commonParser as cmp
from mellolib.readData import MelloDataSet
from mellolib.globalConstants import ARCH
from mellolib.models import transfer
from datetime import datetime

import optuna
from sklearn.externals import joblib

def train(model, loader, criterion, optimizer, epoch, options):
    model.train()
    for batch_idx, (inp, target) in enumerate(loader):
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

        if batch_idx % options.val_frequency == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inp), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))

def test(model, loader, options):
    with torch.no_grad():
        gt, pred = generate_results(loader, options, model)
    accuracy = eval_accuracy(gt,pred)
    return accuracy

def objective(trial, options):
    train_dataset = MelloDataSet(options.train_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=options.shuffle)
    test_dataset = MelloDataSet(options.val_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    criterion = cmp.criterion_selection(options.criterion)

####################### Hyperparameter autotuning session ######################
    '''
    Remember to comment out uneccessary hyperparameter that you are not tuning.
    Each hyperparameter also has a fixed value so that you can comment out the
    optuna version without disrupting the flow
    Available hyperparameters are:
        Final layers of model (model.<final layer name>)
        Learning rate (lr)
        Momentum (momentum)
        Optimizer (optimizer)
    '''

    #----------------------------------------------------------------#
    model = cmp.model_selection(options.arch)
    num_layers = trial.suggest_int('num_layers',1,3)
    layer_list = []
    prev_num_neurons = 512
    for i in range(num_layers):
        num_neurons = int(trial.suggest_loguniform('num_neurons_{}'.format(i),4,prev_num_neurons))
        layer_list.append(('fc_{}'.format(i), nn.Linear(prev_num_neurons,num_neurons)))
        layer_list.append(('relu_{}'.format(i), nn.ReLU()))
        prev_num_neurons = num_neurons

    layer_list.append(('fc_last', nn.Linear(in_features=num_neurons, out_features=2)))
    layer_list.append(('output', nn.Softmax(dim=1)))
    fc = nn.Sequential(OrderedDict(layer_list))
    model.fc = fc
    if (options.deploy_on_gpu):
        if (not torch.cuda.is_available()):
            print("GPU device doesn't exist")
        else:
            model = model.cuda()
            print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    lr = 0.001
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    momentum = 0.9
    # momentum = trial.suggest_uniform('momentum', 0.4, 0.99)
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    optimizer_list = {'SGD': optim.SGD, 'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
    optimizer_name = trial.suggest_categorical('optimizer',['SGD','RMSprop','Adam'])

    optimizer = optimizer_list['Adam'](model.parameters(), lr=lr)
    # if (optimizer_name == 'SGD' or optimizer_name = 'RMSprop'):
    #     optimizer = optimizer_list[optimizer_name](model.parameters(), momentum=momentum, lr=lr)
    # elif (optimizer_name == 'Adam'):
    #     optimizer = optimizer_list['Adam'](model.parameters(), lr=lr)
    # else:
    #     print("Error: please check optimizer_list and optimizer_name")
    #     exit(-1)
    #----------------------------------------------------------------#

################################################################################

    now = datetime.now()
    date = datetime.timestamp(now)
    timestamp = datetime.fromtimestamp(date)
    for epoch in range(options.epoch):
        train(model, train_loader, criterion, optimizer, epoch, options)
        test_accuracy = test(model, test_loader, options)

        if options.checkpoint:
            torch.save(model.state_dict(),options.weight_addr + str(timestamp) + "_epoch" +  str(epoch))

    return test_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cmp.basic_runner(parser)
    # optuna runner is like beefy runner ... but on steroid
    cmp.beefy_runner(parser)
    options = parser.parse_args()

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial: objective(trial, options), n_trials=100)
    joblib.dump(study, options.log_addr)
