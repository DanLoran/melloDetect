# melloDetect
### Running basic_runner
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
The previous command will setup a port connection to http://localhost:8097/. Proceed to that link to open up the plotting platform. Afterward, to run an example of basic_runner.py (remember to set the dataset directory correctly in the run_basic_runner.sh), do:
```
bash run_basic_runner
```
