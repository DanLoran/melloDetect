#/bin/bash

trainAddr="fakeData/trainData/"
testAddr="fakeData/valData/"
weightAddr="weight/"
logAddr="logs/basic_runner.txt"
architecture="tiny_cnn"

python3 basic_runner.py \
--debug=True \
--show-learning-curve=False \
--deploy-on-gpu=False \
--run-validation=False \
--checkpoint=False \
--run-at-checkpoint=False \
--train-addr=${trainAddr} \
--val-addr=${testAddr} \
--weight-addr=${weightAddr} \
--log-addr=${logAddr} \
--arch=${architecture}
