import torch
import argparse
import numpy as np

from visdom import Visdom
from torch.optim import SGD, Adam
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('../')

from mellolib import commonParser as cmp
from mellolib.splitter import Splitter
from mellolib.models import transfer
from mellolib.eval import eval_selection
import mellolib.globalConstants

############################ Setup parser ######################################
parser = argparse.ArgumentParser()
cmp.basic_runner(parser)
cmp.beefy_runner(parser)
options = parser.parse_args()

##################### Setup visdom visualization ###############################
viz = 0
if(options.show_visdom):
    try:
        viz = Visdom()
    except Exception as e:
        print(
            "The visdom experienced an exception while running: {}\n"
            "The demo displays up-to-date functionality with the GitHub "
            "version, which may not yet be pushed to pip. Please upgrade "
            "using `pip install -e .` or `easy_install .`\n"
            "If this does not resolve the problem, please open an issue on "
            "our GitHub.".format(repr(e))
        )

########################## Choose architecture #################################
cmp.DEBUGprint("Loading model. \n", options.debug)
model = cmp.init_model(options)

########################### Environment setup ##################################

# Resume from checkpoint option
cmp.DEBUGprint("Loading previous state. \n", options.debug)
if options.run_at_checkpoint:
    model.load_state_dict(torch.load(options.weight_addr))

# Setup log
cmp.DEBUGprint("Setup log. \n", options.debug)
log = open(options.log_addr,"w+")

# Basic runner stuff
cmp.DEBUGprint("Initialize runner. \n", options.debug)

########################## Training setup ######################################
n_eps = options.epoch
batch_size = options.batch_size
optimizer = cmp.optimizer_selection(options.optimizer, model.parameters(), options.lr, options.momentum)
criterion = cmp.criterion_selection(options.criterion)
dataset_generator = Splitter(options.data_addr, options.split, options.seed,
    pretrained_model=options.pretrained_model,
    debug=options.debug,
    use_sex=options.use_sex,
    positive_case_percent=options.positive_case_percent)
dataset = dataset_generator.generate_training_data()
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=options.shuffle)

batch_n = 0
loop_itr = 0
viz_itr = 0
val_itr = 0
losses = []
viz_time = []
val_time = []

# evaluation parameters
if (options.run_validation):
    test_dataset = dataset_generator.generate_validation_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    eval_score = []

cmp.DEBUGprint("Training... \n", options.debug)


############################# Training loop ####################################
# current date and time
now = datetime.now()

date = datetime.timestamp(now)
timestamp = datetime.fromtimestamp(date).strftime('%Y-%m-%d_%H:%M:%S')
print("Start training at ", timestamp)

# Begin Training (ignore tqdm, it is just a progress bar GUI)
model.train()

try:
    for ep in tqdm(range(n_eps)):
        for inp, target in loader:
            loop_itr += 1
            if mellolib.globalConstants.DEPLOY_ON_GPU:
                target = torch.autograd.Variable(target).cuda()
                inp = torch.autograd.Variable(inp).cuda()
            else:
                target = torch.autograd.Variable(target)
                inp = torch.autograd.Variable(inp)

            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            log.write(str(loss.cpu().detach().numpy().item()))
            log.write("\n")

            if options.show_visdom:
                losses.append(loss.cpu().detach().numpy())
                viz_time.append(viz_itr)
                viz.line(X=viz_time,Y=losses,win='viz1', name="Learning curve",
                opts={'linecolor': np.array([[0, 0, 255],]), 'title':"Learning curve"})
                viz_itr+=1

            if (options.run_validation == True and (loop_itr % options.val_frequency == 0)):
                score, name = eval_selection(test_loader, options, model)
                eval_score.append(score)
                eval_name = name

                # show evaluation
                if options.show_visdom:
                    val_time.append(val_itr)
                    viz.line(X =val_time, Y = eval_score, win='viz2', name=eval_name,
                    opts={'linecolor': np.array([[255, 0, 0],]), 'title':eval_name})
                    val_itr+=1
                else:
                    print(eval_name + str(eval_score))

        if options.checkpoint:
            if (options.run_at_checkpoint):
                dir = options.weight_addr.split('/')
                save_name = ''
                for i in range(len(dir) - 1):
                    save_name += '/'
                    save_name += dir[i]
            else:
                save_name = options.weight_addr
            torch.save(model.state_dict(),save_name + str(timestamp) + "_epoch" +  str(ep))
    log.close()
except KeyboardInterrupt:
    if (options.save_if_interupt):
        print("User interupt, runner will save and gracefully exit now")
        if (options.run_at_checkpoint):
            dir = options.weight_addr.split('/')
            save_name = ''
            for i in range(len(dir) - 1):
                save_name += '/'
                save_name += dir[i]
        else:
            save_name = options.weight_addr
        torch.save(model.state_dict(),save_name + 'interupt_' + str(timestamp))
