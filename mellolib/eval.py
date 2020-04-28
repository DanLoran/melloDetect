import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def eval_auc(test_loader, options, model):
    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()

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

def eval_accuracy(test_loader, options, model):
    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()

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

    # Compute Accuracy
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    total = 0
    corr = 0
    for i in range(gt.shape[0]):
        truth = list(gt[i])
        if (pred[i][0] > pred[i][1]):
            vote = [1,0]
        else:
            vote = [0,1]
        if (truth == vote):
            corr += 1
        total += 1

    return (corr / total) * 100
