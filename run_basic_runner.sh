#/bin/bash

trainAddr="dataset/Data/Images/"
testAddr="dataset/Data/Images/"
weightAddr="weight/"
logAddr="logs/basic_runner.txt"
architecture="trans_shufflenet"

python3 basic_runner.py \
--debug=True \
--show-learning-curve=True \
--deploy-on-gpu=True \
--run-validation=True \
--checkpoint=False \
--run-at-checkpoint=False \
--train-addr=${trainAddr} \
--val-addr=${testAddr} \
--weight-addr=${weightAddr} \
--log-addr=${logAddr} \
--arch=${architecture}
