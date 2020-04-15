import argparse

def basic_runner(parser):
    parser.add_argument("--deploy-on-gpu", action="store_true",
                        help="Run the trainer/validator on the GPU. Default:\
                        true" )

    parser.add_argument("--run-validation", action="store_true",
                        help="Validate result after training. Default:\
                        true" )

    parser.add_argument("--checkpoint", action="store_true",
                        help="Save weights during training. Default:\
                        true")

    parser.add_argument("--run-at-checkpoint", action="store_true",
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

    parser.add_argument("--arch", type = str, choices=["zoo-resnet18"],
                        help="Neural network architecture. Default:\
                        zoo-resnet18" )
