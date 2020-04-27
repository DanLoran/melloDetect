# MelloDetect

## Set up
To setup the environment you will need Docker and make. With both installed, you can use the following commands to execute tasks easily
```
make docker     // create docker image
make run        // run container and starts the visdom server
make exec       // exec into the container with the current directory mounted as working volume
```
Once you are inside the container, you can run the Python scripts as described below.

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

## Running basic_runner
The basic runner run SGD optimization method with loss calculation using BCE
equation. The runner can be executed on GPU. It will run for 10 epochs with
batch size of 32. The Learning rate is set at 0.01. The runner will only consider subset="MALIGNANT" label. The validation metric is ROC AUC score. For more info, do
```
python3 basic_runner.py -h
```
To run a visualization plot for learning curve, do:
```
python3 -m visdom.server
```
The previous command will setup a port connection to http://localhost:8097/. Proceed to that link to open up the plotting platform. Afterward, to run an example of basic_runner.py (remember to set the dataset directory correctly in the run_basic_runner.sh), do on a seperate terminal:
```
bash run_basic_runner
```
