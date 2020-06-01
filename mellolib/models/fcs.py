import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class tiny_fc(nn.Module):
    def __init__(self):
        super(tiny_fc, self).__init__()
        self.fc1 = nn.Linear(256*256*3,1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100,10)
        self.fc4 = nn.Linear(10,2)
        self.sm = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sm(x)
        return x

class resnetFC(nn.Module):
    def __init__(self):
        super(resnetFC, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=512, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class resnetFC2048(nn.Module):
    def __init__(self):
        super(resnetFC2048, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=2048, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x
