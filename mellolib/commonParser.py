import argparse
import torch
from mellolib.models import transfer
from mellolib.globalConstants import ARCH
from mellolib.globalConstants import LOSS
from mellolib.globalConstants import OPTIM

for library in ARCH:
    try:
        if('trans' not in library):
            exec("from mellolib.models.{module} import {module}".format(module=library))
    except Exception as e:
        print(e)

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def DEBUGprint(message, choice):
    if (choice):
        print(message)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def model_selection(choice):
    if choice == "tiny_fc":
        model = tiny_fc()

    elif choice == "tiny_cnn":
        model = tiny_cnn()

    elif choice == "trans_resnet18":
        model = transfer.resnet18()

    elif choice == "trans_mobilenet":
        model = transfer.mobilenet()

    elif choice == "trans_alexnet":
        model = transfer.alexnet()

    elif choice == "trans_vgg":
        model = transfer.vgg()

    elif choice == "trans_densenet":
        model = transfer.densenet()

    elif choice == "trans_inception":
        model = transfer.inception()

    elif choice == "trans_googlenet":
        model = transfer.googlenet()

    elif choice == "trans_shufflenet":
        model = transfer.shufflenet()

    else:
        print("Architecture don't exist!")
        exit(1)

    return model

def optimizer_selection(choice):
    if choice == "SGD":
        optimizer = torch.optim.SGD()

    elif choice == "Adam":
        optimizer = torch.optim.Adam()

    else:
        print("Optimizer don't exist!")
        exit(1)

    return optimizer

def criterion_selection(choice):
    if choice == "BCE":
        criterion = torch.nn.BCELoss()

    else:
        print("Criterion don't exist!")
        exit(1)

    return criterion

def basic_runner(parser):

    parser.add_argument("--file", type=open, action=LoadFromFile)

    parser.add_argument("--show-visdom", type=boolean_string, default=True,
                        help="Plot the stats in realtime. Default:\
                        true" )

    parser.add_argument("--deploy-on-gpu", type=boolean_string, default=True,
                        help="Run the trainer/validator on the GPU. Default:\
                        true" )

    parser.add_argument("--debug", type=boolean_string, default=True,
                        help="Print verbosely. Default: true" )

    parser.add_argument("--run-validation", type=boolean_string, default=True,
                        help="Validate result after training. Default:\
                        true" )

    parser.add_argument("--checkpoint", type=boolean_string, default=True,
                        help="Save weights during training. Default:\
                        true")

    parser.add_argument("--run-at-checkpoint", type=boolean_string, default=True,
                        help="Resume training at checkpoint. Default:\
                        true")

    parser.add_argument("--train-addr", type=str, default="./trainData/",
                        help="Directory where train dataset is stored. Default:\
                        ./trainData/" )

    parser.add_argument("--val-addr", type=str, default="./valData/",
                        help="Directory where validation dataset is stored. \
                        Default:  ./valData/" )

    parser.add_argument("--weight-addr", type=str, default="./weight/",
                        help="Directory where weight will be saved. Default: \
                        ./weight/")

    parser.add_argument("--log-addr", type=str, default="./log/",
                        help="Directory where training log will be saved. \
                        Default: ./log/")

    parser.add_argument("--arch", type = str, choices=ARCH,
                        help="Neural network architecture")

def beefy_runner(parser):
    parser.add_argument("--optimizer", type = str, choices=OPTIM,
                        help="Optimization options")

    parser.add_argument("--lr", type = float, default=0.001,
                        help="Learning rate. Default: 0.001")

    parser.add_argument("--momentum", type = float, default=0.9,
                        help="Momentum. Default: 0.9")

    parser.add_argument("--batch-size", type = int, default=32,
                        help="Batch size. Default: 32")

    parser.add_argument("--epoch", type = int, default=1,
                        help="Number of epochs. Default: 1")

    parser.add_argument("--criterion", type = str, choices=LOSS,
                        help="Loss equations.")

    parser.add_argument("--shuffle", type=boolean_string, default=True,
                        help="Shuffle data during training. Default: true")
