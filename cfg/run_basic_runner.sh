#/bin/bash

trainAddr="./augmented_dataset/Data/"
testAddr="./augmented_dataset/Data/"
weightAddr="weight/"
logAddr="logs/basic_runner.txt"
architecture="trans_resnet18"

python3 basic_runner.py \
--debug=True \
--show-visdom=True \
--deploy-on-gpu=False \
--run-validation=True \
--checkpoint=True \
--run-at-checkpoint=False \
--train-addr=${trainAddr} \
--val-addr=${testAddr} \
--weight-addr=${weightAddr} \
--log-addr=${logAddr} \
--arch=${architecture}
