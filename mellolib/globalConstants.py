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
        'resnet50_fc',
        'efficient_net_b0_fc',
        'efficient_net_b1_fc',
        'efficient_net_b2_fc',
        'efficient_net_b3_fc',
        'efficient_net_b4_fc',
        'efficient_net_b5_fc',
        'efficient_net_b6_fc',
        'efficient_net_b7_fc']

OPTIM = ['SGD','Adam']

LOSS = ['BCE']

EVAL = ['AUC','ACCURACY','F1', 'PRECISION', 'RECALL','TN','TP','FN','FP']

PRETRAINED_MODEL_POOL = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'alexnet', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']

DEPLOY_ON_GPU = None
