import mellolib.globalConstants

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def eval_selection(test_loader, options, model):
    gt, pred = generate_results(test_loader, options, model)
    # evaluate the model
    if options.eval_type == "AUC":
        eval_score = eval_auc(gt, pred)
        eval_name = "AUC Score "

    elif options.eval_type == "ACCURACY":
        eval_score = eval_accuracy(gt, pred)
        eval_name = "Accuracy "

    elif options.eval_type == "F1":
        eval_score = eval_f1(gt, pred)
        eval_name = "F1 Score "

    elif options.eval_type == "PRECISION":
        eval_score = eval_precision(gt, pred)
        eval_name = "Precision Score "

    elif options.eval_type == "RECALL":
        eval_score = eval_recall(gt, pred)
        eval_name = "Recall Score "

    elif options.eval_type == "TN":
        eval_score,_,_,_ = eval_confuse(gt, pred)
        eval_name = "True Negative "

    elif options.eval_type == "TP":
        _,eval_score,_,_ = eval_confuse(gt, pred)
        eval_name = "True Positive "

    elif options.eval_type == "FN":
        _,_,eval_score,_ = eval_confuse(gt, pred)
        eval_name = "False Negative "

    elif options.eval_type == "FP":
        _,_,_,eval_score = eval_confuse(gt, pred)
        eval_name = "False Positive "

    else:
        print("Error: evaluation not implemented!")
        exit(0)

    return eval_score, eval_name

def generate_results(test_loader, options, model):
    # Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
    # and model prediction (pred) information.
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    model.eval()

    for inp, target in test_loader:
        if mellolib.globalConstants.DEPLOY_ON_GPU:
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

    return gt, pred
def eval_auc(gt, pred):
    # Compute the model area under curve (AUC).
    auc = roc_auc_score(gt, pred)
    return auc

def eval_accuracy(gt, pred):
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

def eval_f1(gt, pred):
    gt = np.argmax(gt, axis = 1)
    pred = np.argmax(pred, axis = 1)
    f1 = f1_score(gt, pred)

    return f1

def eval_precision(gt, pred):
    gt = np.argmax(gt, axis = 1)
    pred = np.argmax(pred, axis = 1)
    precision = precision_score(gt, pred)

    return precision

def eval_recall(gt, pred):
    gt = np.argmax(gt, axis = 1)
    pred = np.argmax(pred, axis = 1)
    recall = recall_score(gt, pred)

    return recall

def eval_confuse(gt, pred):
    gt = np.argmax(gt, axis = 1)
    pred = np.argmax(pred, axis = 1)
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tn, tp, fn, fp
