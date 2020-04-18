#/bin/bash

trainAddr="fakeData/trainData/"
testAddr="fakeData/valData/"
weightAddr="./weight/"
architecture="zoo-resnet18"

python3 basic_runner.py \
--deploy-on-gpu=false \
--run-validation=true \
--checkpoint=false \
--run-at-checkpoint=false \
--train-addr=${trainAddr} \
--test-addr=${testAddr} \
--weight-addr=${weightAddr} \
--log-addr=${logAddr} \
--arch=${architecture}
