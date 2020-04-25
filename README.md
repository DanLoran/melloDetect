# MelloDetect

## Set up
To setup the environment you will need Docker and make. With both installed, you can use the following commands to execute tasks easily
```
make docker     // create docker image
make run        // run container and starts the visdom server
make exec       // exec into the container with the current directory mounted as working volume
```
Once you are inside the container, you can run the Python scripts as described below.

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
