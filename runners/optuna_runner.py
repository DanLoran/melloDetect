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

from mellolib.eval import generate_results, eval_auc
from mellolib import commonParser as cmp
from mellolib.splitter import Splitter
from mellolib.models import transfer
import mellolib.globalConstants
from datetime import datetime

import optuna
import joblib

def train(model, loader, criterion, optimizer, epoch, options):
    model.train()
    for batch_idx, (inp, target) in enumerate(loader):
        if mellolib.globalConstants.DEPLOY_ON_GPU:
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

        if batch_idx % 10 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inp), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))

def test(model, loader, options):
    with torch.no_grad():
        gt, pred = generate_results(loader, options, model)
    score = eval_auc(gt,pred) # remeber to change this to a parameter
    return score

def objective(trial, options):
    dataset_generator = Splitter(options.data_addr, options.split, options.seed,
        pretrained_model=options.pretrained_model, debug=options.debug,
        positive_case_percent=options.positive_case_percent)

    train_dataset = dataset_generator.generate_training_data()
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=options.shuffle)

    test_dataset = dataset_generator.generate_validation_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    criterion = cmp.criterion_selection(options.criterion)

####################### Hyperparameter autotuning session ######################
    '''
    Remember to comment out uneccessary hyperparameter that you are not tuning.
    Each hyperparameter also has a fixed value so that you can comment out the
    optuna version without disrupting the flow. Option 1 is fixed value. Option
    2 is autotune.

    Available hyperparameters so far are:
        Final layers of model (model.<final layer name>)
        Learning rate (lr)
        Momentum (momentum)
        Optimizer (optimizer)
    '''

    #------------------------------Final layers--------------------------------#
    model = cmp.init_model(options)
    # Option 1:
    # Nothing here

    # Option 2:
    # num_layers = trial.suggest_int('num_layers',1,3)
    # layer_list = []
    # prev_num_neurons = 512
    # for i in range(num_layers):
    #     num_neurons = int(trial.suggest_loguniform('num_neurons_{}'.format(i),4,prev_num_neurons))
    #     layer_list.append(('fc_{}'.format(i), nn.Linear(prev_num_neurons,num_neurons)))
    #     layer_list.append(('relu_{}'.format(i), nn.ReLU()))
    #     prev_num_neurons = num_neurons
    #
    # layer_list.append(('fc_last', nn.Linear(in_features=num_neurons, out_features=2)))
    # layer_list.append(('output', nn.Softmax(dim=1)))
    # fc = nn.Sequential(OrderedDict(layer_list))
    # model.fc = fc

    #--------------------------------------------------------------------------#

    #-------------------------------Learning rate-------------------------------#
    # Options 1:
    # lr = options.lr_fix

    # Options 2:
    lr = trial.suggest_loguniform('lr', options.lr_lower, options.lr_upper)
    #--------------------------------------------------------------------------#

    #-------------------------------Momentum-----------------------------------#
    # Options 1:
    # momentum = options.momentum_fix

    # Options 2:
    momentum = trial.suggest_uniform('momentum', options.momentum_lower, options.momentum_upper)
    #--------------------------------------------------------------------------#

    #--------------------------------Optimizer---------------------------------#
    optimizer_list = {'SGD': optim.SGD, 'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

    # Options 1:
    # optimizer = optimizer_list['Adam'](model.parameters(), lr=lr)

    # Options 2:
    optimizer_name = trial.suggest_categorical('optimizer',['SGD','RMSprop','Adam'])
    if (optimizer_name == 'SGD' or optimizer_name == 'RMSprop'):
        optimizer = optimizer_list[optimizer_name](model.parameters(), momentum=momentum, lr=lr)
    elif (optimizer_name == 'Adam'):
        optimizer = optimizer_list['Adam'](model.parameters(), lr=lr)
    else:
        print("Error: please check optimizer_list and optimizer_name")
        exit(-1)
    #--------------------------------------------------------------------------#

################################################################################

    now = datetime.now()
    date = datetime.timestamp(now)
    timestamp = datetime.fromtimestamp(date)
    for epoch in range(options.epoch):
        train(model, train_loader, criterion, optimizer, epoch, options)
        test_score = test(model, test_loader, options)

        if options.checkpoint:
            if epoch % options.save_freq == 0:
                print('Saving model!')
                torch.save(model.state_dict(),options.weight_addr + str(timestamp) + "_epoch_" +  str(epoch) + "_score_" + str(test_score))

    return test_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cmp.basic_runner(parser)
    cmp.optuna_runner(parser)
    options = parser.parse_args()

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction=options.direction)
    study.optimize(lambda trial: objective(trial, options), n_trials=options.num_trials)
    joblib.dump(study, options.log_addr)
