import argparse
import torch
from mellolib.models import *
from mellolib.globalConstants import ARCH
from mellolib.globalConstants import LOSS
from mellolib.globalConstants import OPTIM
from mellolib.globalConstants import EVAL
import mellolib.globalConstants

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
        model = fcs.tiny_fc()

    elif choice == "tiny_cnn":
        model = tiny_cnn.tiny_cnn()

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

    elif choice == "resnet18_fc":
        model = fcs.resnetFC()

    elif choice == "resnet50_fc":
        model = fcs.resnetFC2048()

    elif choice == "fc513":
        model = fcs.FC513()

    elif choice == "fc2049":
        model = fcs.FC2049()

    else:
        print("Architecture don't exist!")
        exit(1)

    return model

def init_model(options):
    model = model_selection(options.arch)
    mellolib.globalConstants.DEPLOY_ON_GPU = options.deploy_on_gpu
    if (mellolib.globalConstants.DEPLOY_ON_GPU):
        if (not torch.cuda.is_available()):
            raise EnvironmentError('GPU requested but cuda not available.')
        model = model.cuda()
        print("Deploying model on: " + torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")
    return model


def optimizer_selection(choice, params, lr, momentum):
    if choice == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)

    elif choice == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr)

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

    parser.add_argument("--data-addr", type=str, default="./dataset/Data",
                        help="Directory where all data is stored. Default:\
                        ./dataset/Data" )

    parser.add_argument("--split", type=float, default=0.8,
                        help="Split ratio. Default 0.8 (80% training)")

    parser.add_argument("--seed", type=float, default=123,
                        help="Seed to be used for train/validate split randomness")

    parser.add_argument("--weight-addr", type=str, default="./weight/",
                        help="Directory where weight will be saved. Default: \
                        ./weight/")

    parser.add_argument("--log-addr", type=str, default="./log/",
                        help="Directory where training log will be saved. \
                        Default: ./log/")

    parser.add_argument("--arch", type = str, choices=ARCH,
                        help="Neural network architecture")

    parser.add_argument("--pretrained_model", default=None, type=str,
                        help="The pretrained model to use for feature extraction. \
                         Must be used with classifier as model.")

    parser.add_argument("--use_sex", type=boolean_string, default=False,
                        help="Use sex as additional features.")

def beefy_runner(parser):
    parser.add_argument("--optimizer", type = str, choices=OPTIM,
                        help="Optimization options")

    parser.add_argument("--lr", type = float, default=0.001,
                        help="Learning rate. Default: 0.001")

    parser.add_argument("--momentum", type = float, default=-1.0,
                        help="Momentum. Default: -1.0")

    parser.add_argument("--batch-size", type = int, default=32,
                        help="Batch size. Default: 32")

    parser.add_argument("--epoch", type = int, default=1,
                        help="Number of epochs. Default: 1")

    parser.add_argument("--criterion", type = str, choices=LOSS,
                        help="Loss equations.")

    parser.add_argument("--shuffle", type=boolean_string, default=True,
                        help="Shuffle data during training. Default: true")

    parser.add_argument("--eval-type", type= str, choices=EVAL,
                        help="Evaluation type")

    parser.add_argument("--val-frequency", type = int, default=1,
                        help="Number of training iterations until run evaluation. Default: 1")

    parser.add_argument("--save-if-interupt", type=boolean_string, default=True,
                        help="Save the weight if Ctrl+C interrupt. Default: true")


def eval_runner(parser):
    parser.add_argument("--file", type=open, action=LoadFromFile)

    parser.add_argument("--debug", type=boolean_string, default=True,
                        help="Print verbosely. Default: true" )

    parser.add_argument("--deploy-on-gpu", type=boolean_string, default=True,
                        help="Run the validator on the GPU. Default:\
                        true" )

    parser.add_argument("--eval-weight-addr", type=str, default="./weight/",
                        help="Directory where all weight will be evaluated. Default: \
                        ./weight/")

    parser.add_argument("--split", type=float, default=0.8,
                       help="Split ratio. Default 0.8 (80% training)")

    parser.add_argument("--seed", type=float, default=123,
                       help="Seed to be used for train/validate split randomness")

    parser.add_argument("--data-addr", type=str, default="./valData/",
                        help="Directory where validation dataset is stored. \
                        Default:  ./valData/" )

    parser.add_argument("--arch", type = str, choices=ARCH,
                        help="Neural network architecture")

    parser.add_argument("--pretrained_model", default=None, type=str,
                        help="The pretrained model to use for feature extraction. \
                         Must be used with classifier as model.")

    parser.add_argument("--use_sex", type=boolean_string, default=False,
                        help="Use sex as additional features.")

def optuna_runner(parser):
    parser.add_argument("--num-trials", type=int, default=20,
                        help="Number of trials optuna should execute the runner \
                        Default: 20" )

    parser.add_argument("--lr-fix", type=float, default=0.001,
                        help="Learning rate (if not part of autotuning). \
                        Default: 0.001")

    parser.add_argument("--lr-lower", type=float, default=0.000001,
                        help="Lower bound of learning rate. \
                        Default: 0.000001")

    parser.add_argument("--lr-upper", type=float, default=0.001,
                        help="Upper bound of learning rate. \
                        Default: 0.001")

    parser.add_argument("--momentum-fix", type=float, default=0.001,
                        help="Momentum (if not part of autotuning). \
                        Default: 0.99")

    parser.add_argument("--momentum-lower", type=float, default=0.4,
                        help="Lower bound of momentum. \
                        Default: 0.4")

    parser.add_argument("--momentum-upper", type=float, default=0.99,
                        help="Upper bound of momentum. \
                        Default: 0.99")

    parser.add_argument("--eval-type", type= str, choices=EVAL,
                        help="Evaluation type")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size. Default: 32")

    parser.add_argument("--direction", type=str, default="maximize",
                        help="Autotune direction. Default: maximize")

    parser.add_argument("--save-freq", type=int, default=5,
                        help="Save weights every <--save-freq> epochs. \
                        Default: 5")

    parser.add_argument("--epoch", type = int, default=1,
                        help="Number of epochs. Default: 1")

    parser.add_argument("--shuffle", type=boolean_string, default=True,
                        help="Shuffle data during training. Default: true")

    parser.add_argument("--criterion", type = str, choices=LOSS,
                        help="Loss equations.")

def prediction_runner(parser):
    parser.add_argument("--file", type=open, action=LoadFromFile)

    parser.add_argument("--weight-filepath", type=str,
                        help="Path to the weights of the model used for the \
                        predictions")

    parser.add_argument("--arch", type = str, choices=ARCH,
                        help="Neural network architecture")

    parser.add_argument("--data-addr", type=str, default="./valData/",
                        help="Directory where dataset is stored. \
                        Default:  ./valData/" )

    parser.add_argument("--deploy-on-gpu", type=boolean_string, default=True,
                        help="Run the trainer/validator on the GPU. Default:\
                        true" )

    parser.add_argument("--pretrained_model", default=None, type=str,
                        help="The pretrained model to use for feature extraction. \
                         Must be used with classifier as model.")

    parser.add_argument("--use_sex", type=boolean_string, default=False,
                        help="Use sex as additional features.")
