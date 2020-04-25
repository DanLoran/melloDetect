import argparse

def DEBUGprint(message, opt):
    if (opt):
        print(message)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def basic_runner(parser):
    parser.add_argument("--show-learning-curve", type=boolean_string, default=True,
                        help="Show the Learning curve in realtime. Default:\
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

    parser.add_argument("--arch", type = str, choices=["zoo-resnet18",
                                                       "tiny-fc",
                                                       "tiny-cnn",
                                                       "resnet18"],
                        help="Neural network architecture. Default:\
                        zoo-resnet18" )
