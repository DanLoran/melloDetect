FIELDS = {"BENIGN":0, "MALIGNANT":1}
ARCH = ['tiny_fc',
        'tiny_cnn',
        'trans_resnet18',
        'trans_mobilenet',
        'trans_alexnet',
        'trans_vgg',
        'trans_densenet',
        'trans_inception',
        'trans_googlenet',
        'trans_shufflenet',
        'resnet18_fc']

OPTIM = ['SGD','Adam']

LOSS = ['BCE']

EVAL = ['AUC','ACCURACY','F1', 'PRECISION', 'RECALL','TN','TP','FN','FP']
