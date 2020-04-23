import torch
import torch.nn as nn
import torch.functional as F

class tiny_cnn(nn.Module):
    def __init__(self):
        super(tiny_cnn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias=True)

        self.pool = nn.Maxpool2d(kernel_size=3,
                                 stride=3)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=12,
                               kernel_size=6,
                               stride=3,
                               padding=0,
                               bias=True)

        self.fc1 = nn.Linear(in_features=100,
                             out_features=1000)

        self.fc2 = nn.Linear(in_features=1000,
                             out_features=100)

        self.fc3 = nn.Linear(in_features=100,
                             out_features=10)

        self.fc2 = nn.Linear(in_features=10,
                             out_features=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)
