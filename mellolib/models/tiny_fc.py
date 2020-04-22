import torch
import torch.nn as nn
import torch.nn.functional as F

class tiny_fc(nn.Module):
    def __init__(self):
        super(tiny_fc, self).__init__()
        self.fc1 = nn.Linear(256*256*3,1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100,10)
        self.fc4 = nn.Linear(10,1)

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)
