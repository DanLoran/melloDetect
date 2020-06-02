import torchvision

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
        'resnet18_fc',
        'resnet50_fc']

OPTIM = ['SGD','Adam']

LOSS = ['BCE']

EVAL = ['AUC','ACCURACY','F1', 'PRECISION', 'RECALL','TN','TP','FN','FP']

global PRETRAINED_MODEL_POOL
PRETRAINED_MODEL_POOL = {
    'resnet18': torchvision.models.resnet18(pretrained=True),
    'resnet34': torchvision.models.resnet34(pretrained=True),
    'resnet50': torchvision.models.resnet50(pretrained=True),
    'resnet101': torchvision.models.resnet101(pretrained=True),
    'alexnet': torchvision.models.alexnet(pretrained=True),
}
