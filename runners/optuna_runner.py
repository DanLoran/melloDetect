import torch
import argparse
import numpy as np

import torch.optim as optim
from torch.nn import BCELoss
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
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

def runner(trial):

    parser = argparse.ArgumentParser()
    cmp.basic_runner(parser)
    # optuna runner is like beefy runner ... but on steroid
    cmp.beefy_runner(parser)
    options = parser.parse_args()

    train_dataset = MelloDataSet(options.train_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=options.shuffle)
    test_dataset = MelloDataSet(options.val_addr, transforms=Compose([Resize((256,256)), ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    criterion = cmp.criterion_selection(options.criterion)
    model = cmp.model_selection(options.arch)
    if (options.deploy_on_gpu):
        if (not torch.cuda.is_available()):
            print("GPU device doesn't exist")
        else:
            model = model.cuda()
            print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")

    # Hyperparameter autotuning
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.4, 0.99)
    optimizer = trial.suggest_categorical('optimizer',[optim.SGD, optim.RMSprop, optim.Adam])(model.parameters(), lr=lr)
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
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=runner, n_trials=3)
    joblib.dump(study, '../logs/optuna.pkl')
