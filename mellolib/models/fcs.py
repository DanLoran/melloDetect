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

class FC512(nn.Module):
    def __init__(self):
        super(FC512, self).__init__()
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

class FC2048(nn.Module):
    def __init__(self):
        super(FC2048, self).__init__()
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

class FC1280(nn.Module):
    def __init__(self):
        super(FC1280, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1280, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC1408(nn.Module):
    def __init__(self):
        super(FC1408, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1408, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC1536(nn.Module):
    def __init__(self):
        super(FC1536, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1536, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC1792(nn.Module):
    def __init__(self):
        super(FC1792, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1792, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC2304(nn.Module):
    def __init__(self):
        super(FC2304, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=2304, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC2560(nn.Module):
    def __init__(self):
        super(FC2560, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=2560, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x


class FC513(nn.Module):
    def __init__(self):
        super(FC513, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=513, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x

class FC2049(nn.Module):
    def __init__(self):
        super(FC2049, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=2049, out_features=100)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(in_features=100, out_features=10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(in_features=10, out_features=2)),
            ('output', nn.Softmax(dim=1))
        ]))

    def forward(self,x):
        x = self.fc(x)
        return x
