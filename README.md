# MelloDetect

## Set up
To setup the environment you will need Docker and make. With both installed, you can use the following commands to execute tasks easily
```
make docker     // create docker image
make run        // run container and starts the visdom server
make exec       // exec into the container with the current directory mounted as working volume
```
## :warning:	Downloading data :warning:
To download the ISIC data, do:
```
bash ./dataset/getData.sh
```
DO NOT run the `dataset/getData.sh` inside this repo. Copy the entier `./dataset` folder and put it somewhere with large storage space before downloading. The total space required is aroung 50GB. I recommend using this as using this as your database, never modify, augment or delete it. The images will be available in `Data/Images`

To split the data into training and validation set, do something like:
```
python3 makepartition.py --p 5 --source-addr <src> --train-addr <train> --val-addr <val> --max-pull 5000
```
`--source-addr` is database address.

`--train-addr` is train address.

`--val-addr` is validation address.

`--p` is the percentage of images pulled from database that goes into validation set. (1-p) will go to train set

`--max-pull` is the maximum number of images pulled from the database.

After this, you can run the augmented script if need.

## :warning:	Adding new architectures :warning:
It is crucial to follow the provided steps when adding a new architecture. We want everyone to be able to use the pipeline without any hickups. Inserting your own architecture midway in a runner file may make the runner break, and worse, if you push the runner that has a hard-code architecture in and someone else use it without knowing they are training your model! So don't be lazy :wink:, these steps will isolate your model under development from the rest of the pipeline:

For transfer learning models:
1. Create your model as a function and put it in `mellolib/models/transfer.py` similar to those founds in the file.
2. Append your model name in `mellolib/globalConstants.py` in the `ARCH` list. Your model must have a `trans_` pretext before a unique name. Do not name something that is already in the list. For example, if your model name is `examplenet`, append `trans_examplenet` into `ARCH`.
3. Add an if statement in the `model_selection()` function in `mellolib/commonParser.py` for your model.
4. You can start calling your model in the config files.

For models you wrote from scratch:
1. Create your model as a class and put it in a separate file under `mellolib/models`.
2. Append your model name in `mellolib/globalConstants.py` in the `ARCH` list. Do not name something that is already in the list. For example, if your model name is `examplenet`, append `examplenet` into `ARCH`.
3. Add an if statement in the `model_selection()` function in `mellolib/commonParser.py` for your model.
4. You can start calling your model in the config files.

## Training runners
Runner files are located in `./runners`, they are python script that stitch all mellolib and pytorch routine together to train a model. There are:
1. `basic_runner.py` which can only do a limited range of training, but it is fast to develop/test new architecture. It can also be used as template to develope more sophisticated runners. The runner will train with: Adam optimizer doing Binary Cross Entropy Loss calculation. The runner will run for 10 epochs with batch size of 32 and learning rate of 0.001. The evaluation metric is AUC.
2. `beefy_runner.py` is an extension of `basic_runner.py` that allows you to specify optimizer, learning rate, momentum (if applicable), batch size, epoch number, loss function, shuffle input data, evaluation type, on top of all parameters provided by `basic_runner.py`.

To see the list of parameters, do
```
python3 <runner> -h
```
There are 2 ways of calling the runner. You can do it manually by specifying all required parameters. For example
```
python3 basic_runner.py --debug=True --show-visdom=True --deploy-on-gpu=True --run-validation=True --checkpoint=False --run-at-checkpoint=False --train-addr=/home/minh/git/melloDetect/augmented_dataset/Data/ --val-addr=/home/minh/git/melloDetect/augmented_dataset/Data/ --weight-addr=/home/minh/git/melloDetect/weight/ --log-addr=/home/minh/git/melloDetect/logs/basic_runner.txt --arch=tiny_cnn

```
Or you can put all parameters in a `.cfg` file with each parameter on its own line. For example:
```
python3 basic_runner.p --file ./cfg/basic_runner_example.cfg
```
