#/bin/bash

trainAddr="/home/minh/git/melloDetect/fakeData/trainData/"
testAddr="/home/minh/git/melloDetect/fakeData/valData/"
weightAddr="/home/minh/git/melloDetect/weight/"
logAddr="/home/minh/git/melloDetect/logs/basic_runner.txt"
architecture="zoo-resnet18"

python3 basic_runner.py \
--debug=True \
--deploy-on-gpu=False \
--run-validation=True \
--checkpoint=False \
--run-at-checkpoint=False \
--train-addr=${trainAddr} \
--val-addr=${testAddr} \
--weight-addr=${weightAddr} \
--log-addr=${logAddr} \
--arch=${architecture}
